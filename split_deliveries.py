#!/usr/bin/env python
"""
Interactive DSP splitter.

Reads a delivery export (long OR wide/matrix layout), maps the DSP names it
finds onto the active stores returned by the Revelator API, and writes one
headerless CSV of releaseIds per DSP per status.

Run:
    export REVELATOR_TOKEN="<<token>>"
    python split_deliveries.py deliveries.csv 125555 --upc-map upcs.csv

    # inspect the DSP x status breakdown without splitting (no token needed)
    python split_deliveries.py deliveries.csv 125555 --upc-map upcs.csv --analyze

Input layouts
-------------
LONG  – one row per (upc, dsp, status):
    upc,dsp_name,dsp_status
    18736007781,Spotify,PUBLISHED

WIDE  – one row per upc, one column per DSP, status in the cells:
    UPC,7 Digital,Amazon HD,Spotify
    18736007781,PUBLISHED,TAKEDOWN,PUBLISHED

Comma, tab and semicolon delimiters are detected automatically.
"""

from __future__ import annotations
import os, sys, re, unicodedata, requests, difflib, textwrap
from pathlib import Path
from collections import Counter
from typing import Iterator
import pandas as pd
from tqdm import tqdm


CHUNK_ROWS            = 250_000
TMP_DIR               = Path("/tmp/dsp_split")
OUTPUT_DIR            = Path("./out")
STATUS_OUTPUT_DIRS    = {
    "PUBLISHED": Path("./out"),            # keep existing location for published
    "TAKENDOWN": Path("./out_takendown"),  # new folder for takedowns
}
# Normalised spelling -> canonical status. Add new source spellings here.
STATUS_ALIASES = {
    "PUBLISHED": "PUBLISHED",
    "LIVE":      "PUBLISHED",
    "TAKENDOWN": "TAKENDOWN",
    "TAKEDOWN":  "TAKENDOWN",   # the spelling used by the delivery matrix export
}
LINE_TERM = "\r\n"          # use CRLF line endings in all CSV outputs
STATUS_COLUMN_CANDIDATES = ("dsp_status", "product_status")
UPC_COLUMN_CANDIDATES    = ("upc", "upc_code")
MISSING_UPC_REPORT       = Path("./out_reports/missing_upcs.csv")
ENCODING                 = "utf-8-sig"   # tolerates a BOM from Sheets/Excel exports


# --------------------------------------------------------------------------- #
#  Normalisation helpers                                                      #
# --------------------------------------------------------------------------- #

def safe_slug(name: str) -> str:
    """
    Normalise a name for use in filenames:
    • Strip accents
    • Collapse spaces
    • Uppercase
    • Replace unsafe characters (like slashes) with underscores
    """
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\s+", " ", name).strip().upper()
    name = re.sub(r"[^\w\- ]+", "_", name)  # remove unsafe filename characters
    return name


def slug(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", name).strip().upper()


def normalize_status(value: str) -> str:
    """Uppercase status and strip whitespace/underscores/hyphens for matching."""
    return re.sub(r"[\s_\-]+", "", str(value)).upper()


def canonical_status(value: str) -> str | None:
    """Map a raw cell value onto a canonical status, or None if unrecognised."""
    return STATUS_ALIASES.get(normalize_status(value))


def normalize_upc(value: str) -> str:
    """Canonicalise UPC strings so mapping is consistent."""
    s = str(value).strip()
    if s.endswith(".0"):   # common artefact when numbers are read as floats
        s = s[:-2]
    return s


# --------------------------------------------------------------------------- #
#  Source-format detection                                                    #
# --------------------------------------------------------------------------- #

def detect_delimiter(path: Path) -> str:
    """Guess the delimiter from the header line (tab / semicolon / comma)."""
    with path.open("r", encoding=ENCODING, errors="ignore") as fh:
        header = fh.readline()
    counts = {d: header.count(d) for d in ("\t", ";", ",")}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def read_header(path: Path, sep: str) -> list[str]:
    header = pd.read_csv(path, sep=sep, nrows=0, encoding=ENCODING)
    return [str(c).strip() for c in header.columns]


def find_upc_column(columns: list[str]) -> str | None:
    """Match a UPC column case-insensitively ('UPC', 'upc', 'Upc Code', …)."""
    lowered = {c.lower().replace(" ", "_"): c for c in columns}
    for cand in UPC_COLUMN_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    return None


def find_status_column(columns: list[str]) -> str | None:
    lowered = {c.lower().replace(" ", "_"): c for c in columns}
    for cand in STATUS_COLUMN_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    return None


SAMPLE_ROWS       = 2_000   # rows sampled to decide which columns are DSPs
DSP_COLUMN_CUTOFF = 0.5     # share of sampled cells that must be a known status


def classify_columns(path: Path, sep: str, upc_col: str,
                     candidates: list[str]) -> tuple[list[str], list[dict]]:
    """
    Decide which of the non-UPC columns actually hold delivery statuses.

    A matrix export usually carries extra columns (delivery scope, notes,
    territories…). Melting those in turns their free text into fake statuses,
    so anything whose sampled values are mostly NOT recognised statuses is
    excluded. Returns (dsp_columns, rejected) where rejected carries a reason
    and an example value for reporting.
    """
    sample = pd.read_csv(path, sep=sep, dtype=str, nrows=SAMPLE_ROWS,
                         encoding=ENCODING, on_bad_lines="skip")
    sample.columns = [str(c).strip() for c in sample.columns]

    dsp_columns: list[str] = []
    rejected: list[dict] = []

    for col in candidates:
        if col not in sample.columns:
            continue
        values = sample[col].dropna().map(str).str.strip()
        values = values[values != ""]
        if values.empty:
            rejected.append({"column": col, "reason": "no values in sample",
                             "example": "", "match_rate": 0.0})
            continue
        hits = values.map(lambda v: canonical_status(v) is not None).mean()
        if hits >= DSP_COLUMN_CUTOFF:
            dsp_columns.append(col)
        else:
            example = values.iloc[0]
            rejected.append({
                "column": col,
                "reason": "values are not delivery statuses",
                "example": example[:80] + ("…" if len(example) > 80 else ""),
                "match_rate": round(float(hits), 3),
            })
    return dsp_columns, rejected


def describe_source(path: Path, forced: str = "auto",
                    only_columns: list[str] | None = None,
                    ignore_columns: list[str] | None = None) -> dict:
    """
    Work out how to read the source file.

    Returns a dict with: layout ('long'|'wide'), sep, upc_col, status_col,
    dsp_columns and rejected_columns (wide only).
    """
    sep     = detect_delimiter(path)
    columns = read_header(path, sep)
    # drop the empty columns pandas invents for trailing delimiters
    columns = [c for c in columns if c and not c.startswith("Unnamed:")]

    upc_col    = find_upc_column(columns)
    status_col = find_status_column(columns)

    if upc_col is None:
        raise SystemExit(
            f"❌ No UPC column found. Expected one of "
            f"({', '.join(UPC_COLUMN_CANDIDATES)}); header was: {', '.join(columns[:8])}…"
        )

    layout = forced
    if layout == "auto":
        layout = "long" if status_col else "wide"

    if layout == "long" and status_col is None:
        raise SystemExit(
            f"❌ --format long requires a status column "
            f"({', '.join(STATUS_COLUMN_CANDIDATES)}), none found."
        )

    dsp_columns: list[str] = []
    rejected: list[dict] = []
    if layout == "wide":
        candidates = [c for c in columns if c != upc_col]
        if ignore_columns:
            drop = {c.strip().lower() for c in ignore_columns}
            candidates = [c for c in candidates if c.lower() not in drop]

        if only_columns:
            wanted = {c.strip().lower() for c in only_columns}
            dsp_columns = [c for c in candidates if c.lower() in wanted]
            unknown = wanted - {c.lower() for c in dsp_columns}
            if unknown:
                raise SystemExit(
                    f"❌ --dsp-columns names columns that are not in the header: "
                    f"{', '.join(sorted(unknown))}"
                )
        else:
            dsp_columns, rejected = classify_columns(path, sep, upc_col, candidates)

        if not dsp_columns:
            raise SystemExit(
                "❌ Wide layout detected but no column holds delivery statuses. "
                "Use --dsp-columns to name them explicitly."
            )

    return {
        "layout": layout,
        "sep": sep,
        "upc_col": upc_col,
        "rejected_columns": rejected,
        "status_col": status_col,
        "dsp_columns": dsp_columns,
    }


def iter_long_rows(path: Path, spec: dict) -> Iterator[pd.DataFrame]:
    """
    Stream the source file as normalised long-format chunks with columns
    upc_norm / dsp_name / status_canon / status_raw, whatever the input layout.
    """
    sep     = spec["sep"]
    upc_col = spec["upc_col"]

    if spec["layout"] == "long":
        usecols = [upc_col, "dsp_name", spec["status_col"]]
        reader = pd.read_csv(path, sep=sep, dtype=str, chunksize=CHUNK_ROWS,
                             encoding=ENCODING, on_bad_lines="skip", usecols=usecols)
        for chunk in reader:
            chunk = chunk.rename(columns={upc_col: "_upc",
                                          spec["status_col"]: "_status"})
            yield _finalise(chunk)
    else:
        reader = pd.read_csv(path, sep=sep, dtype=str, chunksize=CHUNK_ROWS,
                             encoding=ENCODING, on_bad_lines="skip")
        for chunk in reader:
            chunk.columns = [str(c).strip() for c in chunk.columns]
            keep = [upc_col] + spec["dsp_columns"]
            chunk = chunk[[c for c in keep if c in chunk.columns]]
            melted = chunk.melt(id_vars=[upc_col],
                                var_name="dsp_name",
                                value_name="_status")
            melted = melted.rename(columns={upc_col: "_upc"})
            melted = melted.dropna(subset=["_status"])
            yield _finalise(melted)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["_upc"]).copy()
    df["upc_norm"]     = df["_upc"].map(normalize_upc)
    df["status_raw"]   = df["_status"].astype(str).str.strip()
    df["status_canon"] = df["_status"].map(canonical_status)
    return df[["upc_norm", "dsp_name", "status_raw", "status_canon"]]


# --------------------------------------------------------------------------- #
#  UPC → releaseId mapping                                                    #
# --------------------------------------------------------------------------- #

def load_upc_mapping(path: Path) -> dict[str, str]:
    """Load UPC → releaseId mapping from the provided CSV."""
    if not path.exists():
        raise SystemExit(f"❌ UPC mapping file not found: {path}")
    sep = detect_delimiter(path)
    try:
        df = pd.read_csv(path, sep=sep, usecols=["upc", "releaseId"],
                         dtype=str, encoding=ENCODING)
    except ValueError as e:
        raise SystemExit(
            f"❌ UPC mapping file must have 'upc' and 'releaseId' columns ({path}): {e}"
        )
    df["upc_norm"] = df["upc"].map(normalize_upc)
    mapping = dict(zip(df["upc_norm"], df["releaseId"]))
    if not mapping:
        raise SystemExit(f"❌ UPC mapping file was empty: {path}")
    return mapping


# --------------------------------------------------------------------------- #
#  Step to get a breakdown of the analysis                                    #
# --------------------------------------------------------------------------- #

def analyze_dsp_breakdown(source_csv: Path, spec: dict) -> None:
    print("\n🔍 Generating DSP × Status Breakdown…\n")
    counts: Counter[tuple[str, str]] = Counter()
    with tqdm(desc="Scanning source", unit="rows") as pbar:
        for chunk in iter_long_rows(source_csv, spec):
            pbar.update(len(chunk))
            counts.update(zip(chunk["dsp_name"], chunk["status_raw"]))

    if not counts:
        print("⚠️  No rows found.")
        return

    tidy = pd.DataFrame(
        [{"dsp_name": d, "status": s, "n": n} for (d, s), n in counts.items()]
    )
    pivot = tidy.pivot_table(index="dsp_name", columns="status",
                             values="n", aggfunc="sum", fill_value=0)
    print("\n✅ DSP-wise Status Breakdown:\n")
    print(pivot)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    out_file = OUTPUT_DIR / "dsp_status_breakdown.csv"
    pivot.to_csv(out_file, lineterminator=LINE_TERM)
    written.append(out_file)

    unknown: Counter[str] = Counter()
    for (_, status), n in counts.items():
        if canonical_status(status) is None:
            unknown[status] += n

    if unknown:
        unk_file = OUTPUT_DIR / "unrecognized_statuses.csv"
        (pd.DataFrame(unknown.most_common(), columns=["value", "occurrences"])
           .to_csv(unk_file, index=False, lineterminator=LINE_TERM))
        written.append(unk_file)
        print(f"\n⚠️  {len(unknown):,} distinct values will be SKIPPED "
              f"(not published/takendown). Top 10:")
        for value, n in unknown.most_common(10):
            shown = value if len(value) <= 70 else value[:70] + "…"
            print(f"   • {shown!r}  ({n:,})")

    if spec.get("rejected_columns"):
        col_file = OUTPUT_DIR / "column_classification.csv"
        pd.DataFrame(spec["rejected_columns"]).to_csv(col_file, index=False,
                                                      lineterminator=LINE_TERM)
        written.append(col_file)

    print("\n📤 Files written:")
    for p in written:
        print(f"   • {p.resolve()}")


# --------------------------------------------------------------------------- #
#  Enterprise guard – the token must belong to the enterprise on the CLI       #
# --------------------------------------------------------------------------- #

# Claim names that have been seen carrying the enterprise, normalised to
# lowercase alphanumerics. Add to this list if your token uses another name.
ENTERPRISE_CLAIM_KEYS = (
    "enterpriseid", "enterprise", "entid",
    "primaryenterpriseid", "currententerpriseid", "enterpriseids",
)


def _normalise_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def decode_jwt_claims(token: str) -> dict | None:
    """
    Decode the payload of a JWT without verifying its signature.

    We are not authenticating anything here – the API does that. This only
    reads the claims so we can compare the enterprise against the CLI value.
    """
    import base64, json

    parts = token.strip().split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)   # restore base64 padding
    try:
        raw = base64.urlsafe_b64decode(payload)
        claims = json.loads(raw)
    except Exception:
        return None
    return claims if isinstance(claims, dict) else None


def find_enterprise_claim(claims: dict,
                          extra_key: str | None = None) -> tuple[str, object] | None:
    """
    Walk the claims looking for an enterprise identifier.

    Returns (claim_path, value) for the first match, or None. Nested objects
    and claims holding embedded JSON are both searched.
    """
    import json

    wanted = set(ENTERPRISE_CLAIM_KEYS)
    if extra_key:
        wanted.add(_normalise_key(extra_key))

    def walk(node, path: str):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else str(k)
                if _normalise_key(k) in wanted and not isinstance(v, (dict, list)):
                    yield here, v
                elif _normalise_key(k) in wanted and isinstance(v, list):
                    yield here, v
                else:
                    yield from walk(v, here)
        elif isinstance(node, str) and node.strip().startswith(("{", "[")):
            try:
                yield from walk(json.loads(node), path)
            except Exception:
                return

    for path, value in walk(claims, ""):
        return path, value
    return None


def verify_enterprise(token: str, enterprise_id: int,
                      claim_key: str | None = None,
                      skip: bool = False) -> None:
    """Abort unless the token's enterprise matches the one given on the CLI."""
    if skip:
        print("⚠️  Enterprise check skipped by request (--skip-enterprise-check).")
        return

    claims = decode_jwt_claims(token)
    if claims is None:
        sys.exit(
            "❌ Could not read the token's claims, so the enterprise cannot be "
            "verified.\n   Re-run with --skip-enterprise-check only if you are "
            "certain the token belongs to enterprise "
            f"{enterprise_id}."
        )

    found = find_enterprise_claim(claims, claim_key)
    if found is None:
        visible = ", ".join(sorted(claims)[:12])
        sys.exit(
            "❌ No enterprise claim found in the token.\n"
            f"   Claims present: {visible}\n"
            "   Point at the right one with --enterprise-claim <name>, or "
            "bypass with --skip-enterprise-check."
        )

    path, value = found
    values = value if isinstance(value, list) else [value]
    values = [str(v).strip() for v in values]

    if str(enterprise_id) not in values:
        sys.exit(
            "❌ ENTERPRISE MISMATCH – aborting before anything is written.\n"
            f"   You asked for : {enterprise_id}\n"
            f"   Token belongs to: {', '.join(values)}  (claim '{path}')\n"
            "   Use the token for the right enterprise, or correct the "
            "enterprise_id argument."
        )

    print(f"🔐  Enterprise verified: token matches {enterprise_id} (claim '{path}').")


# --------------------------------------------------------------------------- #
#  Step 0.  Call Revelator – return only ACTIVE stores                        #
# --------------------------------------------------------------------------- #
def get_active_dsps(token: str) -> dict[str, dict]:
    url = "https://api.revelator.com/common/lookup/stores"
    r   = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    active: dict[str, dict] = {}
    for row in r.json():
        if not row.get("isActive"):
            continue
        active_slug = slug(row["name"])
        active[active_slug] = {"id": row["distributorStoreId"], "name": row["name"]}
    if not active:
        raise RuntimeError("No active DSPs from API – check token / account.")
    return active


# --------------------------------------------------------------------------- #
#  Step 1.  Collect the distinct DSP strings in the source                    #
# --------------------------------------------------------------------------- #
def collect_source_dsps(src: Path, spec: dict) -> list[str]:
    """Wide files list their DSPs in the header, so no scan is needed."""
    if spec["layout"] == "wide":
        return sorted(spec["dsp_columns"], key=str.casefold)

    seen = set()
    with tqdm(desc="Scanning source DSP names", unit="rows") as pbar:
        for chunk in pd.read_csv(src, sep=spec["sep"], usecols=["dsp_name"],
                                 dtype=str, chunksize=CHUNK_ROWS,
                                 encoding=ENCODING, on_bad_lines="skip"):
            seen.update(chunk["dsp_name"].dropna().unique().tolist())
            pbar.update(len(chunk))
    return sorted(seen, key=str.casefold)


# --------------------------------------------------------------------------- #
#  Step 2.  Build an initial mapping suggestion                               #
# --------------------------------------------------------------------------- #
def suggest_mapping(source_dsps: list[str], active_dsps: dict[str, dict]) -> dict[str, str | None]:
    active_names      = [meta["name"] for meta in active_dsps.values()]
    mapping: dict[str, str | None] = {}
    for s in source_dsps:
        s_slug = slug(s)
        if s_slug in active_dsps:                       # perfect slug match
            mapping[s] = active_dsps[s_slug]["name"]
            continue
        # fuzzy match – grab the best candidate
        candidates = difflib.get_close_matches(s, active_names, n=1, cutoff=0.6)
        mapping[s] = candidates[0] if candidates else None
    return mapping


# --------------------------------------------------------------------------- #
#  Step 3.  Console‑based reconciliation loop                                 #
# --------------------------------------------------------------------------- #
def interactive_reconcile(mapping: dict[str, str | None],
                          active_dsps: dict[str, dict]) -> dict[str, str | None]:
    """Let the user confirm or edit the mapping.

    Returns the *final* mapping; entries with value None are treated as 'skip'.
    """
    active_names = sorted({m["name"] for m in active_dsps.values()}, key=str.casefold)

    while True:
        # --- show current status ------------------------------------------------
        print("\n========== MAPPING PREVIEW ==========")
        for i, (src, tgt) in enumerate(mapping.items(), 1):
            print(f"{i:2}. {src:<40} →  {tgt or '<< UNMAPPED >>'}")
        user = input(
            "\nType CONFIRMED to proceed, or a comma‑separated list of line numbers to edit: "
        ).strip()

        if user.upper() == "CONFIRMED":
            # we now allow unmapped values – just warn
            skipped = [k for k, v in mapping.items() if v is None]
            if skipped:
                print(
                    f"⚠️  {len(skipped)} DSP(s) will be skipped because they are unmapped:"
                )
                for s in skipped:
                    print(f"   • {s}")
            return mapping

        # --- editing path -------------------------------------------------------
        if not re.fullmatch(r"\d+(?:\s*,\s*\d+)*", user):
            print("⚠️  Enter CONFIRMED or valid line numbers (e.g. 3 or 3,5,7).")
            continue

        for num in map(int, user.split(",")):
            if num < 1 or num > len(mapping):
                print(f"Line {num} is out of range.")
                continue

            src_name = list(mapping.keys())[num - 1]
            print(f"\nMapping for: {src_name}")
            close = difflib.get_close_matches(src_name, active_names, n=7, cutoff=0.3)
            menu = dict(zip("abcdefg", close))
            for k, v in menu.items():
                print(f"  {k}. {v}")
            print("  m. manual entry")
            print("  u. LEAVE UNMAPPED")

            choice = input("Select option: ").strip().lower()

            # ----- leave unmapped ------------------------------------------------
            if choice == "u":
                mapping[src_name] = None
                continue

            # ----- manual entry --------------------------------------------------
            if choice == "m":
                print(
                    "\nActive DSP names (from API):\n"
                    + textwrap.fill(", ".join(active_names), width=80)
                )
                while True:
                    manual = input(
                        'Type exact DSP name, or "BACK" to return, or "EXIT" to abort: '
                    ).strip()
                    if manual.upper() == "EXIT":
                        print("Exiting at user request.")
                        sys.exit(1)
                    elif manual.upper() == "BACK":
                        print("↩️  Returning to mapping menu (no changes made).")
                        break  # do not change the current mapping
                    elif slug(manual) in active_dsps:
                        mapping[src_name] = manual
                        break
                    else:
                        print("⚠️  Name not found. Try again, or type BACK or EXIT.")
                continue

            # ----- close‑match selection -----------------------------------------
            if choice in menu:
                mapping[src_name] = menu[choice]
            else:
                print("Invalid selection – nothing changed for this line.")


# --------------------------------------------------------------------------- #
#  Step 4.  Stream‑split CSV using the *confirmed* mapping                    #
# --------------------------------------------------------------------------- #
def split_csv_by_mapping(src: Path,
                         spec: dict,
                         mapping: dict[str, str | None],
                         active_dsps: dict[str, dict],
                         upc_map: dict[str, str]) -> tuple[list[Path], Path | None]:
    TMP_DIR.mkdir(exist_ok=True, parents=True)
    status_tmp_dirs = {status: TMP_DIR / status.lower() for status in STATUS_OUTPUT_DIRS}
    for d in status_tmp_dirs.values():
        d.mkdir(exist_ok=True, parents=True)
        for stale in d.glob("*.tmp"):      # never append onto a previous run
            stale.unlink()
    for out_dir in STATUS_OUTPUT_DIRS.values():
        out_dir.mkdir(exist_ok=True, parents=True)

    missing_upcs: Counter[str] = Counter()
    skipped_statuses: Counter[str] = Counter()

    # ---- before streaming starts ---------------------------------------------
    duplicates: dict[str, list[str]] = {}
    for dsp_src, tgt in mapping.items():
        if tgt is None:
            continue
        duplicates.setdefault(tgt, []).append(dsp_src)

    for tgt, srcs in duplicates.items():
        if len(srcs) > 1:
            print(f"🔸  {len(srcs)} source DSP names map to '{tgt}': {', '.join(srcs)}")

    # quick‑lookup helpers
    map_to_slug = {src: slug(tgt) for src, tgt in mapping.items() if tgt}
    buffers: dict[tuple[str, str], object] = {}

    with tqdm(desc="Processing CSV", unit="rows") as pbar:
        for chunk in iter_long_rows(src, spec):
            pbar.update(len(chunk))

            unknown = chunk[chunk["status_canon"].isna()]
            if not unknown.empty:
                skipped_statuses.update(unknown["status_raw"].value_counts().to_dict())
            chunk = chunk.dropna(subset=["status_canon"])
            if chunk.empty:
                continue

            # replace dsp_name with canonical slug via mapping
            chunk = chunk.assign(dsp_slug=chunk["dsp_name"].map(map_to_slug))
            chunk = chunk.dropna(subset=["dsp_slug"])
            if chunk.empty:
                continue

            chunk = chunk.assign(release_id=chunk["upc_norm"].map(upc_map))
            missing_mask = chunk["release_id"].isna()
            if missing_mask.any():
                missing_upcs.update(chunk.loc[missing_mask, "upc_norm"].value_counts().to_dict())
            chunk = chunk.dropna(subset=["release_id"])
            if chunk.empty:
                continue

            for (status_norm, canon), grp in chunk.groupby(["status_canon", "dsp_slug"]):
                if canon not in active_dsps:
                    # mapping error should never happen after confirmation
                    continue
                fh = buffers.get((status_norm, canon))
                if fh is None:
                    safe_canon = safe_slug(canon)
                    tmp_dir = status_tmp_dirs[status_norm]
                    fh = (tmp_dir / f"{safe_canon}.tmp").open("a", encoding="utf-8")
                    buffers[(status_norm, canon)] = fh
                grp["release_id"].to_csv(fh, header=False, index=False,
                                         lineterminator=LINE_TERM)

    for fh in buffers.values():
        fh.close()

    outputs: list[Path] = []
    for status_norm, out_dir in STATUS_OUTPUT_DIRS.items():
        tmp_dir = status_tmp_dirs[status_norm]
        for canon, meta in active_dsps.items():
            safe_canon = safe_slug(canon)
            tmp = tmp_dir / f"{safe_canon}.tmp"

            if not tmp.exists():
                continue
            safe_name = safe_slug(meta["name"])
            dest = out_dir / f"{safe_name}_{meta['id']}_deliveries.csv"
            (pd.read_csv(tmp, header=None, names=["release_id"], dtype=str)
                .drop_duplicates()
                .sort_values("release_id")
                .to_csv(dest, header=False, index=False, lineterminator=LINE_TERM))
            outputs.append(dest)
            tmp.unlink(missing_ok=True)

    if skipped_statuses:
        print("\nℹ️  Rows skipped because their status is neither published nor takendown:")
        for status, n in skipped_statuses.most_common(10):
            print(f"   • {status!r}: {n:,}")

    missing_report: Path | None = None
    if missing_upcs:
        missing_report = MISSING_UPC_REPORT
        missing_report.parent.mkdir(parents=True, exist_ok=True)
        (pd.DataFrame(sorted(missing_upcs.items()), columns=["upc", "occurrences"])
           .to_csv(missing_report, index=False, lineterminator=LINE_TERM))
    return outputs, missing_report


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Delivery Split Agent")
    parser.add_argument("source_csv", type=Path, help="Path to input CSV")
    parser.add_argument("enterprise_id", type=int, help="Enterprise ID")
    parser.add_argument("--analyze", action="store_true", help="Show DSP breakdown only")
    parser.add_argument("--upc-map", type=Path, required=True,
                        help="CSV mapping UPC → releaseId (columns: upc, releaseId)")
    parser.add_argument("--format", choices=("auto", "long", "wide"), default="auto",
                        help="Force the input layout instead of detecting it")
    parser.add_argument("--dsp-columns", default=None,
                        help="Comma-separated list of the DSP columns (wide files); "
                             "skips auto-detection")
    parser.add_argument("--ignore-columns", default=None,
                        help="Comma-separated columns to exclude outright (wide files)")
    parser.add_argument("--enterprise-claim", default=None,
                        help="Name of the token claim holding the enterprise, if the "
                             "usual ones are not present")
    parser.add_argument("--skip-enterprise-check", action="store_true",
                        help="Proceed even though the token's enterprise could not be "
                             "matched against enterprise_id (use with care)")
    args = parser.parse_args()

    split_list = lambda s: [p.strip() for p in s.split(",") if p.strip()] if s else None

    if not args.source_csv.exists():
        sys.exit(f"❌ File not found: {args.source_csv}")

    spec    = describe_source(args.source_csv, args.format,
                              only_columns=split_list(args.dsp_columns),
                              ignore_columns=split_list(args.ignore_columns))
    upc_map = load_upc_mapping(args.upc_map)

    sep_label = {"\t": "TAB", ",": "COMMA", ";": "SEMICOLON"}[spec["sep"]]
    print(f"🧭  Detected layout: {spec['layout'].upper()} (delimiter: {sep_label})")
    print(f"🏷️  Using UPC column: {spec['upc_col']}")
    if spec["layout"] == "long":
        print(f"📑  Using status column: {spec['status_col']}")
    else:
        print(f"📊  {len(spec['dsp_columns'])} DSP columns: "
              f"{', '.join(spec['dsp_columns'][:6])}"
              f"{'…' if len(spec['dsp_columns']) > 6 else ''}")
        for r in spec.get("rejected_columns", []):
            print(f"🚫  Excluded column {r['column']!r} – {r['reason']}"
                  + (f" (e.g. {r['example']!r})" if r["example"] else ""))
    print(f"🗂️  Loaded {len(upc_map):,} UPC → releaseId mappings from {args.upc_map}")

    if args.analyze:
        analyze_dsp_breakdown(args.source_csv, spec)
        return

    token = os.getenv("REVELATOR_TOKEN")
    if not token:
        sys.exit("❌ Set REVELATOR_TOKEN environment variable first.")

    # Run full agent – verify the token belongs to this enterprise first, so a
    # wrong token aborts before any store lookup or file is written.
    verify_enterprise(token, args.enterprise_id,
                      claim_key=args.enterprise_claim,
                      skip=args.skip_enterprise_check)

    print("🔑  Fetching active DSPs …")
    active_dsps = get_active_dsps(token)
    print(f"    → {len(active_dsps)} active stores retrieved.")

    print("🔍  Collecting DSP names from source …")
    source_dsps = collect_source_dsps(args.source_csv, spec)
    print(f"    → Found {len(source_dsps)} unique DSP strings in file.")

    mapping   = suggest_mapping(source_dsps, active_dsps)
    confirmed = interactive_reconcile(mapping, active_dsps)

    print("\n👍  Mapping confirmed – starting split.")
    print("📂  Target statuses:",
          ", ".join(f"{s} → {d}" for s, d in STATUS_OUTPUT_DIRS.items()))
    outputs, missing_report = split_csv_by_mapping(args.source_csv, spec, confirmed,
                                                   active_dsps, upc_map)

    print("\n✅  Completed. Generated files:")
    for p in outputs:
        print(f"   • {p}")
    if not outputs:
        print("⚠️  No rows matched criteria; no files written.")
    if missing_report:
        print(f"\n⚠️  {missing_report.name} generated for UPCs without a releaseId mapping: {missing_report}")


if __name__ == "__main__":
    main()