#!/usr/bin/env python3
"""
convert_skyrim_manifest.py

Convert the original YAML manifests in
    original_skyrim_data/{train.yaml,test.yaml}
to the fine-tune-ready JSON-lines manifests

    dataset/train.json
    dataset/test.json

Usage
-----
python convert_skyrim_manifest.py \
    --yaml-root original_skyrim_data \
    --audio-root original_skyrim_data/audio \
    --out-root  dataset
"""

from __future__ import annotations
import argparse, json, pathlib, sys

try:
    import yaml  # PyYAML
except ImportError as exc:  # pragma: no cover
    sys.exit(
        "PyYAML is required (`pip install pyyaml`). "
        f"Import error: {exc}"
    )


def load_yaml(path: pathlib.Path) -> list[dict]:
    """Return the list under the `lines:` key."""
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return doc.get("lines", [])


def make_example(rec: dict, audio_root: pathlib.Path) -> dict:
    """Map one YAML record → training-manifest record."""
    audio_path = audio_root / rec["FileName"]       # absolute/relative—up to you
    duration_sec = round(rec["DurationMs"] / 1000, 3)

    return {
        "audio": {"path": str(audio_path)},
        "sentence": rec["Transcription"].strip(),
        # The dataset is purely English, so we can omit "language".
        # No timestamp labels available, so we also omit "sentences".
        "duration": duration_sec,
    }


def convert_split(
    split_name: str,
    yaml_root: pathlib.Path,
    audio_root: pathlib.Path,
    out_root: pathlib.Path,
) -> None:
    in_path = yaml_root / f"{split_name}.yaml"
    out_path = out_root / f"{split_name}.json"

    records = load_yaml(in_path)

    out_root.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            json.dump(make_example(rec, audio_root), fh, ensure_ascii=False)
            fh.write("\n")  # jsonlines

    print(f"✓ Wrote {len(records):,} examples to {out_path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="YAML → JSON-lines converter")
    p.add_argument("--yaml-root", type=pathlib.Path, required=True,
                   help="Directory containing train.yaml / test.yaml")
    p.add_argument("--audio-root", type=pathlib.Path, required=True,
                   help="Directory where the .wav files live")
    p.add_argument("--out-root",  type=pathlib.Path, default=pathlib.Path("dataset"),
                   help="Where to write train.json / test.json")
    args = p.parse_args(argv)

    for split in ("train", "test"):
        convert_split(split, args.yaml_root, args.audio_root, args.out_root)


if __name__ == "__main__":
    main()
