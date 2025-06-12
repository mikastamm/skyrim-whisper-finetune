#!/usr/bin/env python3
"""
merge_commonvoice_other.py

Scan *all* sub-directories of <root>/commonvoice/**/  that look like
    <lang-or-variant>/
        ├─ clips/
        ├─ other.tsv
        └─ clip_durations.tsv
Combine every other-split row into ONE json-lines file:
    <root>/commonvoice/other.json

Usage
-----
python merge_commonvoice_other.py --root dataset/commonvoice
"""

from __future__ import annotations
import argparse, csv, json, pathlib, sys

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def load_durations(dur_file: pathlib.Path) -> dict[str, float]:
    """clip_durations.tsv ➜ {filename → seconds}."""
    table: dict[str, float] = {}
    try:
        with dur_file.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                ms = int(row["duration[ms]"])
                table[row["clip"]] = round(ms / 1000, 3)
    except FileNotFoundError:
        print(f"[WARN] {dur_file} missing — durations will be absent", file=sys.stderr)
    return table


def each_other_row(tsv_path: pathlib.Path):
    try:
        with tsv_path.open("r", encoding="utf-8") as f:
            yield from csv.DictReader(f, delimiter="\t")
    except FileNotFoundError:
        return  # silently yield nothing


def make_example(
    row: dict,
    *,
    clips_dir: pathlib.Path,
    durations: dict[str, float],
) -> dict:
    fname = row["path"]
    obj = {
        "audio": {"path": str(clips_dir / fname)},
        "sentence": row["sentence"].strip(),
        "language": row.get("locale") or "und",
    }
    if fname in durations:
        obj["duration"] = durations[fname]
    return obj


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Merge all Common Voice other.tsv files into one other.json"
    )
    ap.add_argument(
        "--root",
        type=pathlib.Path,
        required=True,
        help="Top-level directory that contains the per-language folders (e.g. dataset/commonvoice)",
    )
    args = ap.parse_args(argv)

    out_path = args.root / "other.json"
    total = 0

    # Open once for streaming output
    with out_path.open("w", encoding="utf-8") as out_f:

        # walk immediate sub-folders (en/, de/, …)
        for subdir in sorted(p for p in args.root.iterdir() if p.is_dir()):
            tsv_path  = subdir / "other.tsv"
            dur_path  = subdir / "clip_durations.tsv"
            clips_dir = subdir / "clips"

            if not tsv_path.exists():
                continue  # skip folders without other.tsv

            durations = load_durations(dur_path)

            for row in each_other_row(tsv_path):
                json.dump(
                    make_example(row, clips_dir=clips_dir, durations=durations),
                    out_f,
                    ensure_ascii=False,
                )
                out_f.write("\n")
                total += 1

            print(f"✓ merged {subdir.name}/other.tsv ({len(durations):,} durations)")

    print(f"\n=== Wrote {total:,} rows to {out_path} ===")


if __name__ == "__main__":
    main()
