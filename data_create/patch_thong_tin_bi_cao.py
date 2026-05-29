import os
import json
import argparse
from pathlib import Path

def process_files(target_dir: str, source_dir: str):
    target_path = Path(target_dir)
    source_path = Path(source_dir)
    
    if not target_path.exists():
        print(f"Target directory {target_dir} does not exist.")
        return
        
    updated_count = 0
    missing_in_source = 0
    empty_in_source = 0
    
    for file_path in target_path.glob("*.json"):
        if not file_path.is_file():
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[WARN] Error reading {file_path.name}: {e}")
                continue
                
        thong_tin = data.get("THONG_TIN_CHUNG", {})
        if not isinstance(thong_tin, dict):
            thong_tin = {}
            
        bi_cao_list = thong_tin.get("Thong_Tin_Bi_Cao")
        if bi_cao_list is None:
            bi_cao_list = data.get("Thong_Tin_Bi_Cao")
            
        # Check if missing or empty
        if not bi_cao_list or len(bi_cao_list) == 0:
            source_file = source_path / file_path.name
            if not source_file.exists():
                missing_in_source += 1
                continue
                
            with open(source_file, "r", encoding="utf-8") as f:
                try:
                    source_data = json.load(f)
                except Exception as e:
                    print(f"[WARN] Error reading source {source_file.name}: {e}")
                    continue
                    
            source_thong_tin = source_data.get("THONG_TIN_CHUNG", {})
            source_bi_cao_list = None
            if isinstance(source_thong_tin, dict):
                source_bi_cao_list = source_thong_tin.get("Thong_Tin_Bi_Cao")
            if source_bi_cao_list is None:
                source_bi_cao_list = source_data.get("Thong_Tin_Bi_Cao")
                
            if source_bi_cao_list and len(source_bi_cao_list) > 0:
                # Copy the field and maintain order inside THONG_TIN_CHUNG
                ordered_thong_tin = {}
                
                # Try to keep Ma_Ban_An first if it exists
                if "Ma_Ban_An" in thong_tin:
                    ordered_thong_tin["Ma_Ban_An"] = thong_tin["Ma_Ban_An"]
                
                # Insert Thong_Tin_Bi_Cao
                ordered_thong_tin["Thong_Tin_Bi_Cao"] = source_bi_cao_list
                
                # Add any remaining keys
                for k, v in thong_tin.items():
                    if k not in ("Ma_Ban_An", "Thong_Tin_Bi_Cao"):
                        ordered_thong_tin[k] = v
                
                data["THONG_TIN_CHUNG"] = ordered_thong_tin
                
                # If Thong_Tin_Bi_Cao was previously at the root level, remove it
                if "Thong_Tin_Bi_Cao" in data:
                    del data["Thong_Tin_Bi_Cao"]
                
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
                updated_count += 1
                print(f"[INFO] Patched Thong_Tin_Bi_Cao for {file_path.name}")
            else:
                empty_in_source += 1
                
    print("\n[SUMMARY]")
    print(f"Successfully updated files: {updated_count}")
    print(f"Source file not found: {missing_in_source}")
    print(f"Source file also has empty Thong_Tin_Bi_Cao: {empty_in_source}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch missing Thong_Tin_Bi_Cao fields.")
    parser.add_argument("--target-dir", default="data_create/filled_template_openai")
    parser.add_argument("--source-dir", default="/home/hieujayce/Downloads/filled_Chuong_XXII")
    args = parser.parse_args()
    
    process_files(args.target_dir, args.source_dir)
