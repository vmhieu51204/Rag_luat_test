import os
import shutil
from pathlib import Path

def main():
    repo_root = Path("/home/hieujayce/Downloads/complete_repo")
    test_dir = repo_root / "chunk" / "Chuong_XXII_chunked" / "test"
    split_dir = repo_root / "chunk" / "Chuong_XXII_chunked" / "synth" / "split"
    yet_dir = repo_root / "yet"

    # Create target directory
    yet_dir.mkdir(parents=True, exist_ok=True)

    # Get filenames in split_dir
    split_files = set()
    if split_dir.exists():
        split_files = {f.name for f in split_dir.glob("*.json")}

    # Get filenames in test_dir and compare
    copied_count = 0
    if test_dir.exists():
        for test_file_path in test_dir.glob("chunked_*.json"):
            # Get original name by removing "chunked_" prefix
            original_name = test_file_path.name[len("chunked_"):]
            
            if original_name not in split_files:
                target_path = yet_dir / test_file_path.name
                shutil.copy2(test_file_path, target_path)
                copied_count += 1

    print(f"Total files in split: {len(split_files)}")
    print(f"Files copied to 'yet': {copied_count}")

if __name__ == "__main__":
    main()
