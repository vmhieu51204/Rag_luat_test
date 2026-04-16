import os

source_dir = "data_create/Chuong_XXII"
target_dir = "data_create/extracted_fields"

if not os.path.exists(source_dir):
    print(f"Source directory {source_dir} does not exist.")
    exit(1)
if not os.path.exists(target_dir):
    print(f"Target directory {target_dir} does not exist.")
    exit(1)

source_files = set(os.listdir(source_dir))
target_files = os.listdir(target_dir)

deleted_count = 0
for filename in target_files:
    if filename in source_files:
        file_path = os.path.join(target_dir, filename)
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

print(f"Deleted {deleted_count} files from {target_dir} that were already in {source_dir}.")
