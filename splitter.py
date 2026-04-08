import os
import shutil
import random
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# In-memory split (used by pipeline.py)
# ---------------------------------------------------------------------------

def split_train_test(
    docs: list[dict],
    chunks: list[dict],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Split at document level so no doc straddles train/test.
    Returns (train_docs, test_docs, train_chunks, test_chunks).
    """
    rng = random.Random(seed)
    shuffled = docs[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * test_ratio))
    test_ids  = {d["doc_id"] for d in shuffled[:n_test]}
    train_ids = {d["doc_id"] for d in shuffled[n_test:]}

    train_docs   = [d for d in docs   if d["doc_id"] in train_ids]
    test_docs    = [d for d in docs   if d["doc_id"] in test_ids]
    train_chunks = [c for c in chunks if c["doc_id"] in train_ids]
    test_chunks  = [c for c in chunks if c["doc_id"] in test_ids]

    print(f"  Split → train: {len(train_docs)} docs / {len(train_chunks)} chunks"
          f"  |  test: {len(test_docs)} docs / {len(test_chunks)} chunks")
    return train_docs, test_docs, train_chunks, test_chunks

# ---------------------------------------------------------------------------
# File copying split (Standalone execution)
# ---------------------------------------------------------------------------

def split_and_copy_files(input_dir: str, train_dir: str, test_dir: str, test_ratio: float, seed: int):
    """
    Reads JSON files from input_dir, shuffles them, and copies them to train_dir and test_dir.
    """
    input_path = Path(input_dir)
    train_path = Path(train_dir)
    test_path = Path(test_dir)

    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Directory '{input_dir}' does not exist or is not a directory.")
        return

    json_files = sorted(list(input_path.glob("*.json")))
    if not json_files:
        print(f"No JSON files found in '{input_dir}'.")
        return

    rng = random.Random(seed)
    shuffled_files = json_files[:]
    rng.shuffle(shuffled_files)

    n_test = max(1, round(len(shuffled_files) * test_ratio))
    test_files = shuffled_files[:n_test]
    train_files = shuffled_files[n_test:]

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(json_files)} files. Splitting into {len(train_files)} train and {len(test_files)} test files.")

    for f in train_files:
        shutil.copy2(f, train_path / f.name)
    
    for f in test_files:
        shutil.copy2(f, test_path / f.name)

    print(f"Copied {len(train_files)} files to {train_dir}")
    print(f"Copied {len(test_files)} files to {test_dir}")

def main():
    parser = argparse.ArgumentParser(description="Split JSON files into train and test folders.")
    parser.add_argument("--input_dir", required=True, help="Folder with raw JSON files")
    parser.add_argument("--train_dir", required=True, help="Folder to output train JSON files")
    parser.add_argument("--test_dir", required=True, help="Folder to output test JSON files")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Fraction of documents held out as test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")

    args = parser.parse_args()

    split_and_copy_files(
        input_dir=args.input_dir,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
#/splitter.py --input_dir chunk/Chuong_XXII_chunked --train_dir chunk/Chuong_XXII_chunked/train --test_dir chunk/Chuong_XXII_chunked/test