import os
import json

def patch_json_files():
    base_dir = "/home/hieujayce/Downloads/complete_repo/chunk/Chuong_XXII_chunked"
    split_dir = os.path.join(base_dir, "synth", "split")

    if not os.path.exists(split_dir):
        print(f"Directory not found: {split_dir}")
        return

    patched_count = 0
    skipped_count = 0

    for filename in os.listdir(split_dir):
        if not filename.endswith(".json"):
            continue

        split_file_path = os.path.join(split_dir, filename)
        chunked_file_path = os.path.join(base_dir, f"chunked_{filename}")

        if not os.path.exists(chunked_file_path):
            print(f"Warning: {chunked_file_path} does not exist. Skipping {filename}.")
            skipped_count += 1
            continue

        try:
            with open(chunked_file_path, 'r', encoding='utf-8') as f:
                chunked_data = json.load(f)
            
            with open(split_file_path, 'r', encoding='utf-8') as f:
                split_data = json.load(f)

            if "Cac_Dieu_Quyet_Dinh" in chunked_data:
                split_data["Cac_Dieu_Quyet_Dinh"] = chunked_data["Cac_Dieu_Quyet_Dinh"]
                with open(split_file_path, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, ensure_ascii=False, indent=4)
                patched_count += 1
            else:
                print(f"Warning: 'Cac_Dieu_Quyet_Dinh' not found in {chunked_file_path}")
                skipped_count += 1

        except json.JSONDecodeError as e:
             print(f"Error decoding JSON in {filename} or its chunked version: {e}")
             skipped_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped_count += 1

    print(f"Done! Patched {patched_count} files, skipped {skipped_count} files.")

if __name__ == "__main__":
    patch_json_files()
