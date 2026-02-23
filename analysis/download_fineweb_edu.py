"""
Download a subset of FineWeb-Edu to disk for reuse.

Example:
  python -m analysis.download_fineweb_edu --config sample-10BT --num-examples 500000 --output-dir data/fineweb-edu-500k
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu subset to disk")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", type=str, default="sample-10BT")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-examples", type=int, default=500000)
    parser.add_argument("--output-dir", type=str, default="data/fineweb-edu-500k")
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_examples and args.num_examples > 0:
        split = f"{args.split}[:{args.num_examples}]"
    else:
        split = args.split

    print(f"Loading {args.dataset} ({args.config}) split={split}")
    ds = load_dataset(args.dataset, args.config, split=split, cache_dir=args.cache_dir)
    print(f"Loaded {len(ds)} examples. Saving to {output_dir}...")
    ds.save_to_disk(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
