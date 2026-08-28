# DSP Delivery Splitter

Turns a delivery export into one CSV of Revelator `releaseId`s per DSP per status,
ready to feed into a delivery or takedown job.

The main script is `split_deliveries.py`. It reads a source export, asks the
Revelator API which stores are active, has you confirm how the DSP names in your
file map onto those stores, then streams the file out into per-DSP lists.

---

## Requirements

- Python 3.10 or newer (the code uses `X | None` type syntax)
- `pip install pandas requests tqdm`
- A Revelator API token in the `REVELATOR_TOKEN` environment variable

```bash
export REVELATOR_TOKEN="<<your token>>"
```

The token is only needed for the real split. `--analyze` runs offline.

---

## Files in this repo

| File                     | What it is                                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `split_deliveries.py`    | The splitter. This is the one you run.                                                                                                                                   |
| `get_dsp_names.py`       | Throwaway helper: prints a DSP × status pivot for one hardcoded export path. `split_deliveries.py --analyze` supersedes it.                                              |
| `orbital_upc_checker.py` | One-off audit that compares UPC coverage across a fixed set of Orbital spreadsheets. Unrelated to the splitter; expects its input files to sit in the working directory. |
| `apple.xml`              | Sample Apple/iTunes package XML kept for reference. Not used by any script.                                                                                              |
| `.gitignore`             | Ignores `*.csv`, so exports and outputs never get committed.                                                                                                             |

---

## Input files

You need **two** CSVs. Neither has a default path — both are passed explicitly.

### 1. The source export

Two layouts are supported and the script detects which one you have from the
header. Comma, tab and semicolon delimiters are all handled, as is a UTF-8 BOM
from Sheets or Excel.

**Wide (delivery matrix).** One row per UPC, one column per DSP, status in the
cells. This is what a Google Sheets delivery matrix export looks like:

```csv
UPC,7 Digital,Adaptr,Amazon HD,Spotify,Delivery Scope
18736007781,PUBLISHED,PUBLISHED,TAKEDOWN,PUBLISHED,All accounts except: Amazon HD
18736016073,PUBLISHED,PUBLISHED,TAKEDOWN,PUBLISHED,Send only to: Spotify
```

Extra non-DSP columns like `Delivery Scope` are fine. The script samples the
first 2,000 rows and keeps a column as a DSP only if at least half its values
parse as a known status; everything else is excluded and reported by name.

**Long.** One row per UPC/DSP pair:

```csv
upc,dsp_name,dsp_status
18736007781,Spotify,PUBLISHED
18736007781,Amazon Music HD,TAKEDOWN
```

The UPC column can be `upc` or `upc_code`, matched case-insensitively. The
status column can be `dsp_status` or `product_status`, with `dsp_status`
winning if both exist.

**Recognised statuses:** `PUBLISHED` and `LIVE` route to published;
`TAKENDOWN`, `TAKEDOWN` and `TAKEN DOWN` route to takedown. Case, spaces,
underscores and hyphens are all ignored. Anything else (`PENDING`,
`IN REVIEW`, …) is dropped and counted. Add new spellings to `STATUS_ALIASES`
near the top of the script.

### 2. The UPC → releaseId map

Headers must be exactly `upc` and `releaseId`. Extra columns are ignored.

```csv
upc,releaseId
18736007781,4412903
18736016073,4412904
```

Any UPC in the source that isn't in this map is dropped from the output and
logged, because the delivery lists contain releaseIds, not UPCs.

---

## Usage

Always start with a dry run. It needs no token, tells you how the file was
parsed, and writes the breakdown to disk:

```bash
python split_deliveries.py "TuStreams Delivery Matrix.csv" 125555 \
    --upc-map tustreams_upcs.csv \
    --analyze
```

Check three things in that output before going further:

1. The detected layout and delimiter.
2. The `🚫 Excluded column` lines. If a real DSP is listed there, name the DSP
   set yourself with `--dsp-columns`.
3. The skipped-status summary. Statuses you expected to see routed should not
   be sitting in `out/unrecognized_statuses.csv`.

Then the real run:

```bash
export REVELATOR_TOKEN="<<token>>"
python split_deliveries.py "TuStreams Delivery Matrix.csv" 125555 \
    --upc-map tustreams_upcs.csv
```

### Arguments

| Argument                         | Required | Purpose                                                                                        |
| -------------------------------- | -------- | ---------------------------------------------------------------------------------------------- |
| `source_csv`                     | yes      | Path to the export.                                                                            |
| `enterprise_id`                  | yes      | The enterprise this run is for. Checked against the token before anything happens — see below. |
| `--upc-map`                      | yes      | Path to the UPC → releaseId CSV.                                                               |
| `--analyze`                      | no       | Print and export the DSP × status breakdown, then stop.                                        |
| `--format auto\|long\|wide`      | no       | Override layout detection.                                                                     |
| `--dsp-columns "A,B,C"`          | no       | Name the DSP columns explicitly (wide files); skips auto-detection.                            |
| `--ignore-columns "Notes,Scope"` | no       | Drop named columns before detection runs.                                                      |
| `--enterprise-claim <name>`      | no       | Name of the token claim holding the enterprise, if the usual ones aren't present.              |
| `--skip-enterprise-check`        | no       | Proceed when the enterprise can't be read from the token. Use sparingly.                       |

### The enterprise guard

`REVELATOR_TOKEN` must belong to the enterprise named in `enterprise_id`. The
script decodes the token's claims and compares them before it calls the stores
API or creates any directory, so pairing a token with the wrong enterprise fails
loudly and harmlessly:

```
❌ ENTERPRISE MISMATCH – aborting before anything is written.
   You asked for : 125555
   Token belongs to: 999888  (claim 'enterpriseId')
```

Only the claims are read — the signature isn't verified, since the API does that
on the actual request. This is a guard against operator error, not an
authentication step.

Claim names searched: `enterpriseId`, `enterprise`, `entId`,
`primaryEnterpriseId`, `currentEnterpriseId`, `enterpriseIds`. Matching ignores
case, underscores and hyphens; nested objects, claims holding a list of
enterprises, and claims holding embedded JSON are all handled. If your token
uses a different name, pass `--enterprise-claim`, or add it permanently to
`ENTERPRISE_CLAIM_KEYS` in the script.

`--analyze` doesn't need a token and doesn't run the check.

### The mapping step

Once the active stores come back, you get a numbered list pairing each DSP name
in your file with its best match. Exact matches are used directly; the rest are
fuzzy-matched, and a name with no plausible match shows as `<< UNMAPPED >>`.

- Type `CONFIRMED` to proceed. Unmapped names are skipped, with a warning.
- Type line numbers (`3` or `3,5,7`) to fix specific rows.
- Within a row: `a`–`g` picks a suggestion, `m` types an exact name, `u` leaves
  it unmapped.

A 39-column matrix means 39 rows to review. Names like `iTunes MFiT`,
`Touch Tunes` and `YouTube Combined` are the usual ones needing manual help.

---

## Outputs

| Path                                           | Contents                                                                                      |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `out/<DSP>_<storeId>_deliveries.csv`           | Published releaseIds for that store. Headerless, one ID per line, deduplicated, sorted, CRLF. |
| `out_takendown/<DSP>_<storeId>_deliveries.csv` | Same, for takedowns.                                                                          |
| `out_reports/missing_upcs.csv`                 | UPCs with no releaseId in the map, with occurrence counts.                                    |
| `out/dsp_status_breakdown.csv`                 | `--analyze` pivot of DSP × status.                                                            |
| `out/unrecognized_statuses.csv`                | `--analyze` list of skipped status values.                                                    |
| `out/column_classification.csv`                | `--analyze` list of excluded columns and why.                                                 |

Scratch files go to `/tmp/dsp_split` and are cleared at the start of each run.

---

## Troubleshooting

**"No UPC column found."** The header has no `upc` or `upc_code`. If your column
is named something else, rename it, or add the name to `UPC_COLUMN_CANDIDATES`.

**Hundreds of nonsense values in the skipped-status list.** A free-text column
is being read as a DSP. Look at the `🚫 Excluded column` lines; if the offender
isn't listed, exclude it with `--ignore-columns`.

**A DSP got no output file.** Either it was left unmapped at the confirmation
step, its store is inactive in the API response, or every one of its UPCs is
missing from the UPC map. Check `out_reports/missing_upcs.csv` first.

**"ENTERPRISE MISMATCH."** The token belongs to a different enterprise than the
one you passed. Nothing was written. Fix whichever of the two is wrong.

**"No enterprise claim found in the token."** The token stores the enterprise
under a name the script doesn't know. The error lists the claims it can see —
pass the right one with `--enterprise-claim`, then add it to
`ENTERPRISE_CLAIM_KEYS` so the next person doesn't hit it.

**"Could not read the token's claims."** The token isn't a JWT, so the
enterprise can't be checked at all. `--skip-enterprise-check` bypasses this, but
only do that when you've confirmed the token's owner another way.

**"No active DSPs from API."** The token is expired or scoped to the wrong
account.

**Output counts look doubled.** Shouldn't happen — temp files are wiped per run
— but if you interrupt a run mid-write, delete `/tmp/dsp_split` before retrying.
