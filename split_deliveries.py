#!/usr/bin/env python
"""
Interactive DSP splitter.

Run:
    export REVELATOR_TOKEN="<<token>>"
    python split_deliveries_interactive.py deliveries.csv 125555
"""

from __future__ import annotations
import os, sys, re, unicodedata, requests, difflib, textwrap
from pathlib import Path
from collections import Counter
import pandas as pd
from tqdm import tqdm


CHUNK_ROWS            = 250_000
TMP_DIR               = Path("/tmp/dsp_split")
OUTPUT_DIR            = Path("./out")
STATUS_OUTPUT_DIRS    = {
    "PUBLISHED": Path("./out"),            # keep existing location for published
    "TAKENDOWN": Path("./out_takendown"),  # new folder for takedowns
}
LINE_TERM = "\r\n"          # use CRLF line endings in all CSV outputs
STATUS_COLUMN_CANDIDATES = ("dsp_status", "product_status")
UPC_COLUMN_CANDIDATES    = ("upc", "upc_code")
MISSING_UPC_REPORT       = Path("./out_reports/missing_upcs.csv")



# --------------------------------------------------------------------------- #
#  Normalisation helpers                                                      #
# --------------------------------------------------------------------------- #

import re

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
    return re.sub(r"[\\s_\\-]+", "", str(value)).upper()


def normalize_upc(value: str) -> str:
    """Canonicalise UPC strings so mapping is consistent."""
    s = str(value).strip()
    if s.endswith(".0"):   # common artefact when numbers are read as floats
        s = s[:-2]
    return s


def detect_status_column(source_csv: Path) -> str:
    """Return which status column the file exposes (prefer dsp_status)."""
    header = pd.read_csv(source_csv, nrows=0)
    for col in STATUS_COLUMN_CANDIDATES:
        if col in header.columns:
            return col
    raise SystemExit(
        f"❌ None of the expected status columns found "
        f"({', '.join(STATUS_COLUMN_CANDIDATES)})"
    )


def detect_upc_column(source_csv: Path) -> str:
    """Return the UPC column to use (prefer 'upc', fallback to 'upc_code')."""
    header = pd.read_csv(source_csv, nrows=0)
    for col in UPC_COLUMN_CANDIDATES:
        if col in header.columns:
            return col
    raise SystemExit(
        f"❌ None of the expected UPC columns found "
        f"({', '.join(UPC_COLUMN_CANDIDATES)})"
    )


def load_upc_mapping(path: Path) -> dict[str, str]:
    """Load UPC → releaseId mapping from the provided CSV."""
    if not path.exists():
        raise SystemExit(f"❌ UPC mapping file not found: {path}")
    df = pd.read_csv(path, usecols=["upc", "releaseId"], dtype=str)
    df["upc_norm"] = df["upc"].map(normalize_upc)
    mapping = dict(zip(df["upc_norm"], df["releaseId"]))
    if not mapping:
        raise SystemExit(f"❌ UPC mapping file was empty: {path}")
    return mapping

# --------------------------------------------------------------------------- #
#  Step to get a breakdown of the analysis                                    #
# --------------------------------------------------------------------------- #

def analyze_dsp_breakdown(source_csv: Path, status_column: str) -> None:
    print("\n🔍 Generating DSP × Status Breakdown…\n")
    try:
        df = pd.read_csv(source_csv, usecols=["dsp_name", status_column])
        pivot = df.pivot_table(
            index="dsp_name",
            columns=status_column,
            aggfunc="size",
            fill_value=0
        )
        print("✅ DSP-wise Status Breakdown:\n")
        print(pivot)

        # Optionally export to disk
        out_file = OUTPUT_DIR / "dsp_status_breakdown.csv"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pivot.to_csv(out_file,lineterminator=LINE_TERM)
        print(f"\n📤 Breakdown exported to: {out_file}")

    except Exception as e:
        print(f"⚠️  Failed to analyze breakdown: {e}")


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
#  Step 1.  Scan source CSV once to collect the distinct DSP strings          #
# --------------------------------------------------------------------------- #
def collect_source_dsps(src: Path) -> list[str]:
    # minimise memory by reading a single column
    seen = set()
    with tqdm(desc="Scanning source DSP names", unit="rows") as pbar:
        for chunk in pd.read_csv(src, usecols=["dsp_name"],
                                 dtype_backend="pyarrow",
                                 chunksize=CHUNK_ROWS,
                                 on_bad_lines="skip"):
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
        # fuzzy match – grab up to three best candidates
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
                         mapping: dict[str, str],
                         active_dsps: dict[str, dict],
                         status_column: str,
                         upc_column: str,
                         upc_map: dict[str, str]) -> tuple[list[Path], Path | None]:
    TMP_DIR.mkdir(exist_ok=True, parents=True)
    status_tmp_dirs = {status: TMP_DIR / status.lower() for status in STATUS_OUTPUT_DIRS}
    for d in status_tmp_dirs.values():
        d.mkdir(exist_ok=True, parents=True)
    for out_dir in STATUS_OUTPUT_DIRS.values():
        out_dir.mkdir(exist_ok=True, parents=True)
    missing_upcs: Counter[str] = Counter()

    # ---- before streaming starts ---------------------------------------------
    duplicates = {}
    for dsp_src, tgt in mapping.items():            # use a distinct name
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
        for chunk in pd.read_csv(
                src,
                chunksize=CHUNK_ROWS,
                dtype_backend="pyarrow",
                on_bad_lines="skip",
        ):
            pbar.update(len(chunk))
            chunk["_status_norm"] = chunk[status_column].map(normalize_status)
            chunk = chunk[chunk["_status_norm"].isin(STATUS_OUTPUT_DIRS.keys())]
            if chunk.empty:
                continue
            # replace dsp_name with canonical slug via mapping
            chunk["dsp_slug"] = chunk["dsp_name"].map(map_to_slug)
            chunk = chunk.dropna(subset=["dsp_slug"])
            chunk["upc_norm"] = chunk[upc_column].map(normalize_upc)
            chunk["release_id"] = chunk["upc_norm"].map(upc_map)

            missing_mask = chunk["release_id"].isna()
            if missing_mask.any():
                missing_upcs.update(chunk.loc[missing_mask, "upc_norm"].value_counts().to_dict())
            chunk = chunk.dropna(subset=["release_id"])

            for (status_norm, canon), grp in chunk.groupby(["_status_norm", "dsp_slug"]):
                if canon not in active_dsps:
                    # mapping error should never happen after confirmation
                    continue
                fh = buffers.get((status_norm, canon))
                if fh is None:
                    safe_canon = safe_slug(canon)
                    tmp_dir = status_tmp_dirs[status_norm]
                    fh = (tmp_dir / f"{safe_canon}.tmp").open("a", encoding="utf‑8")
                    buffers[(status_norm, canon)] = fh
                grp["release_id"].to_csv(fh, header=False, index=False, lineterminator=LINE_TERM)

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
            (pd.read_csv(tmp, header=None, names=["release_id"])
                .drop_duplicates()
                .sort_values("release_id")
                .to_csv(dest, header=False, index=False, lineterminator=LINE_TERM))
            outputs.append(dest)
            tmp.unlink(missing_ok=True)
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
    parser.add_argument("--upc-map", type=Path, default=Path("orbital_upcs_202512130445.csv"),
                        help="CSV mapping UPC → releaseId (default: orbital_upcs_202512130445.csv)")
    args = parser.parse_args()

    if not args.source_csv.exists():
        sys.exit(f"❌ File not found: {args.source_csv}")

    status_column = detect_status_column(args.source_csv)
    upc_column    = detect_upc_column(args.source_csv)
    upc_map       = load_upc_mapping(args.upc_map)
    print(f"📑  Using status column: {status_column}")
    print(f"🏷️  Using UPC column: {upc_column}")
    print(f"🗂️  Loaded {len(upc_map):,} UPC → releaseId mappings from {args.upc_map}")

    token = os.getenv("REVELATOR_TOKEN")
    if not token:
        sys.exit("❌ Set REVELATOR_TOKEN environment variable first.")

    if args.analyze:
        analyze_dsp_breakdown(args.source_csv, status_column)
        return

    # Run full agent
    print(f"🔐  Running as enterprise: {args.enterprise_id}")

    print("🔑  Fetching active DSPs …")
    active_dsps = get_active_dsps(token)
    print(f"    → {len(active_dsps)} active stores retrieved.")

    print("🔍  Collecting DSP names from source …")
    source_dsps = collect_source_dsps(args.source_csv)
    print(f"    → Found {len(source_dsps)} unique DSP strings in file.")

    mapping = suggest_mapping(source_dsps, active_dsps)
    confirmed = interactive_reconcile(mapping, active_dsps)

    print("\n👍  Mapping confirmed – starting split.")
    print("📂  Target statuses:",
          ", ".join(f"{s} → {d}" for s, d in STATUS_OUTPUT_DIRS.items()))
    outputs, missing_report = split_csv_by_mapping(args.source_csv, confirmed, active_dsps,
                                                   status_column, upc_column, upc_map)

    print("\n✅  Completed. Generated files:")
    for p in outputs:
        print(f"   • {p}")
    if not outputs:
        print("⚠️  No rows matched criteria; no files written.")
    if missing_report:
        print(f"\n⚠️  {missing_report.name} generated for UPCs without a releaseId mapping: {missing_report}")

if __name__ == "__main__":
    main()



# #!/usr/bin/env python
# """
# split_deliveries.py  –  Generate per‑DSP delivery CSVs from a massive source file.

# Usage
# -----
#     export REVELATOR_TOKEN="<< your JWT / OAuth token >>"
#     python split_deliveries.py deliveries.csv 125555

#     • deliveries.csv    – the big input file
#     • 125555            – Revelator “EnterpriseId” (customer)
#     • Outputs are written to a new ./out/ directory

# Requires
# --------
#     pip install pandas pyarrow requests tqdm python-dotenv
# """

# from __future__ import annotations
# import os, sys, re, unicodedata, requests
# from pathlib import Path
# import pandas as pd
# from tqdm import tqdm


# ### ------------------------------------------------------------------------ #
# #  Settings – tweak if you need different behaviour                          #
# ### ------------------------------------------------------------------------ #
# CHUNK_ROWS               = 250_000          # rows per read_csv chunk
# PRODUCT_STATUS_FILTER    = "PUBLISHED"      # only keep these rows
# TMP_DIR                  = Path("/tmp/dsp_split")  # scratch space
# OUTPUT_DIR               = Path("./out")    # final CSVs


# ### ------------------------------------------------------------------------ #
# #  Helpers                                                                   #
# ### ------------------------------------------------------------------------ #
# def slug(name: str) -> str:
#     """Return a canonical, accent‑stripped, upper‑case version of a store name."""
#     name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
#     return re.sub(r"\s+", " ", name).strip().upper()


# def get_active_dsps(token: str, enterprise_id: int) -> dict[str, dict]:
#     """
#     Query Revelator API for active DSPs of this enterprise.

#     Returns
#     -------
#     { canonical_name : { "id": int, "display_name": str } }
#     """
#     url     = "https://app-api.revelator.com/common/lookup/stores"
#     headers = {"Authorization": f"Bearer {token}"}
#     resp    = requests.get(url, headers=headers, timeout=30)
#     resp.raise_for_status()

#     active = {}
#     for row in resp.json():
#         if not row.get("isActive"):
#             continue
#         active[slug(row["name"])] = {
#             "id":   row["distributorStoreId"],
#             "name": row["name"],
#         }
#     if not active:
#         raise RuntimeError("No active DSPs returned – check enterprise ID or token.")
#     return active


# def split_csv_by_dsp(
#     source_csv: Path,
#     active_dsps: dict[str, dict],
#     chunk_rows: int = CHUNK_ROWS,
# ) -> list[Path]:
#     """
#     Stream‑process `source_csv` and write one output file per active DSP.

#     Returns a list of generated Path objects.
#     """
#     TMP_DIR.mkdir(exist_ok=True, parents=True)
#     OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

#     # One temp file handle per DSP, opened lazily
#     buffers: dict[str, object] = {}

#     # Iterate through the big CSV in chunks so we never exhaust RAM
#     with tqdm(desc="Streaming CSV", unit="rows") as pbar:
#         for chunk in pd.read_csv(
#                 source_csv,
#                 chunksize=chunk_rows,
#                 dtype_backend="pyarrow",
#                 on_bad_lines="skip",
#         ):
#             pbar.update(len(chunk))

#             # 1️⃣  Filter early by product_status
#             chunk = chunk[chunk["product_status"] == PRODUCT_STATUS_FILTER]
#             if chunk.empty:
#                 continue

#             # 2️⃣  Normalise dsp_name → canonical slug
#             chunk["dsp_slug"] = chunk["dsp_name"].map(slug)

#             # 3️⃣  Group by slug and append UPCs to that DSP’s temp file
#             for canon, group in chunk.groupby("dsp_slug"):
#                 if canon not in active_dsps:        # Skip unsupported DSPs
#                     continue
#                 buf = buffers.get(canon)
#                 if buf is None:
#                     tmp_path = TMP_DIR / f"{canon}.tmp"
#                     buf = tmp_path.open("a", encoding="utf‑8")
#                     buffers[canon] = buf
#                 group["upc_code"].to_csv(buf, index=False, header=False)

#     # Close all temp handles
#     for buf in buffers.values():
#         buf.close()

#     # 4️⃣  Post‑process each temp file: deduplicate and write final CSV
#     outputs: list[Path] = []
#     for canon, meta in active_dsps.items():
#         tmp_path = TMP_DIR / f"{canon}.tmp"
#         if not tmp_path.exists():
#             continue  # no rows for this DSP

#         dest = OUTPUT_DIR / f"{meta['name']}_{meta['id']}_deliveries.csv"
#         (pd.read_csv(tmp_path, header=None, names=["upc_code"])
#            .drop_duplicates()
#            .sort_values("upc_code")
#            .to_csv(dest, index=False, header=False)
#         )
#         outputs.append(dest)
#         tmp_path.unlink(missing_ok=True)

#     return outputs


# ### ------------------------------------------------------------------------ #
# #  Main                                                                      #
# ### ------------------------------------------------------------------------ #
# def main() -> None:
#     if len(sys.argv) < 3:
#         sys.exit("Usage: python split_deliveries.py <source_csv> <enterprise_id>")

#     source_csv   = Path(sys.argv[1]).expanduser()
#     enterprise_id: int = int(sys.argv[2])

#     if not source_csv.exists():
#         sys.exit(f"Input file not found: {source_csv}")

#     token = os.getenv("REVELATOR_TOKEN")
#     if not token:
#         sys.exit("Set REVELATOR_TOKEN in the environment first.")

#     print("🔎  Discovering active DSPs …")
#     active_dsps = get_active_dsps(token, enterprise_id)
#     print(f"   → Found {len(active_dsps)} active store(s).")

#     print(f"🚚  Processing: {source_csv}")
#     outputs = split_csv_by_dsp(source_csv, active_dsps)
#     print("✅  Generated files:")
#     for p in outputs:
#         print(f"   • {p}")

#     if not outputs:
#         print("⚠️  No rows matched your criteria; nothing written.")


# if __name__ == "__main__":
#     main()
