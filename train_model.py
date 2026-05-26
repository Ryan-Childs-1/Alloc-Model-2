from __future__ import annotations

import argparse
from pathlib import Path

from excel_bridge import read_xlsb_values_pyxlsb
from model_utils import train_models_from_frames
from schema import load_schema


def main():
    parser = argparse.ArgumentParser(description="Train Allocation AI v2 from historical .xlsb allocation workbooks.")
    parser.add_argument("files", nargs="+", help="Historical .xlsb files with human-approved Final Alloc values.")
    parser.add_argument("--schema", default="feature_schema.json")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional per-file row cap for fast tests.")
    args = parser.parse_args()
    schema = load_schema(args.schema)
    frames = []
    for f in args.files:
        print(f"Reading {f}...")
        frames.append(read_xlsb_values_pyxlsb(f, schema["main_sheet"], schema["header_row"], schema["first_data_row"], max_rows=args.max_rows))
    print("Training models...")
    meta = train_models_from_frames(frames, schema_path=args.schema, model_dir=args.model_dir)
    print("Done.")
    for k, v in meta.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
