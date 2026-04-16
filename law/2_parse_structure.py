import re
import json

class Stage2_Parser:
    def __init__(self):
        self.chapter_pattern = re.compile(r'^Chương\s*(?:([IVXLCDM]+)(?:\s+|-|:|\.|$))?(.*)', re.IGNORECASE)
        self.dieu_pattern = re.compile(r'^Điều\s+(\d+[a-zđ]*)\.?\s*(.*)', re.IGNORECASE)
        self.khoan_pattern = re.compile(r'^(\d{1,2})\.(?!\d)\s*(.*)')
        self.diem_pattern = re.compile(r'^([a-zđ])\)(?:\d+)?\s*(.*)')

    def process(self, lines):
        document_data = []
        current_chuong = current_dieu = current_khoan = current_diem = None

        for text in lines:
            chapter_match = self.chapter_pattern.match(text)
            if chapter_match:
                index = chapter_match.group(1).upper() if chapter_match.group(1) else "Unknown"
                current_chuong = {"type": "Chương", "index": index, "text": chapter_match.group(2).strip(" -.:\t"), "Điều": []}
                document_data.append(current_chuong)
                current_dieu = current_khoan = current_diem = None
                continue

            dieu_match = self.dieu_pattern.match(text)
            if dieu_match:
                if current_chuong is None:
                    current_chuong = {"type": "Chương", "index": "Unknown", "text": "", "Điều": []}
                    document_data.append(current_chuong)
                    
                current_dieu = {"type": "Điều", "index": dieu_match.group(1), "text": dieu_match.group(2), "Khoản": []}
                current_chuong["Điều"].append(current_dieu)
                current_khoan = current_diem = None
                continue

            khoan_match = self.khoan_pattern.match(text)
            if khoan_match and current_dieu is not None:
                current_khoan = {"type": "Khoản", "index": khoan_match.group(1), "text": khoan_match.group(2), "Điểm": []}
                current_dieu["Khoản"].append(current_khoan)
                current_diem = None
                continue

            diem_match = self.diem_pattern.match(text)
            if diem_match and current_khoan is not None:
                current_diem = {"type": "Điểm", "index": diem_match.group(1), "text": diem_match.group(2)}
                current_khoan["Điểm"].append(current_diem)
                continue

            if current_diem is not None:
                current_diem["text"] += "\n" + text
            elif current_khoan is not None:
                current_khoan["text"] += "\n" + text
            elif current_dieu is not None:
                current_dieu["text"] += "\n" + text
            elif current_chuong is not None:
                current_chuong["text"] += "\n" + text

        return document_data

if __name__ == "__main__":
    input_file = r"C:\Users\hungn\Downloads\darkin\stage1_lines.json" 
    output_file = r"C:\Users\hungn\Downloads\darkin\stage2_structure.json"
    
    try:
        print("Stage 2: Structuring lines into JSON hierarchy...")
        with open(input_file, 'r', encoding='utf-8') as f:
            clean_lines = json.load(f)
            
        parser = Stage2_Parser()
        structured_data = parser.process(clean_lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=4)
            
        print(f"✅ Success! Saved hierarchical JSON to {output_file}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")