import docx
import re
import json
import copy

import copy
import re

def extract_mentions(text):
    # Fixed regex to safely match characters with boundaries
    pattern = r'\b(?:các\s+)?(Chương|Điều|Khoản|Điểm)\s+((?:(?:\b[IVXLCDM]+\b|\b[0-9]+[a-zđ]*\b|\b[a-zđ]\b)(?:\s*,\s*|\s+và\s+|\s+hoặc\s+|\s+-\s+)*)+)'
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    raw_blocks = []
    for match in matches:
        m_type = match.group(1).capitalize()
        items_str = match.group(2)
        
        if m_type == "Chương":
            ids = [x.upper() for x in re.findall(r'\b([IVXLCDM]+|[0-9]+)\b', items_str, re.IGNORECASE)]
        elif m_type in ["Điều", "Khoản"]:
            ids = re.findall(r'\b([0-9]+[a-zđ]*)\b', items_str, re.IGNORECASE)
        else:
            clean_str = re.sub(r'\b(và|hoặc)\b', '', items_str, flags=re.IGNORECASE)
            ids = re.findall(r'\b([a-zđ])\b', clean_str, re.IGNORECASE)
            
        raw_blocks.append({
            "type": m_type,
            "ids": ids,
            "start": match.start(),
            "end": match.end()
        })
        
    rank = {"Điểm": 1, "Khoản": 2, "Điều": 3, "Chương": 4}
    results = []
    
    i = 0
    while i < len(raw_blocks):
        # Create a deep copy of the block so we can safely mutate it during merging
        chain = [{
            "type": raw_blocks[i]['type'],
            "ids": list(raw_blocks[i]['ids']),
            "start": raw_blocks[i]['start'],
            "end": raw_blocks[i]['end']
        }]
        j = i + 1
        
        while j < len(raw_blocks):
            gap_text = text[chain[-1]['end']:raw_blocks[j]['start']]
            # NEW: Added "và", "hoặc", and "-" to safely bridge items like "khoản 3 và khoản 4"
            gap_valid = re.fullmatch(r'\s*(của|thuộc|tại|,|và|hoặc|-)?\s*', gap_text, re.IGNORECASE)
            
            if not gap_valid:
                break
                
            current_rank = rank.get(chain[-1]['type'], 0)
            next_rank = rank.get(raw_blocks[j]['type'], 0)
            
            if next_rank == current_rank:
                # SAME RANK (e.g. Khoản 3 -> Khoản 4): Merge their IDs together!
                chain[-1]['ids'].extend(raw_blocks[j]['ids'])
                chain[-1]['end'] = raw_blocks[j]['end']  # Extend the gap detection window
                j += 1
            elif next_rank > current_rank:
                # HIGHER RANK (e.g. Khoản -> Điều): Add a new link to the hierarchy chain
                chain.append({
                    "type": raw_blocks[j]['type'],
                    "ids": list(raw_blocks[j]['ids']),
                    "start": raw_blocks[j]['start'],
                    "end": raw_blocks[j]['end']
                })
                j += 1
            else:
                # LOWER RANK (e.g. Điều -> Khoản): Break the chain
                break
                
        def build_hierarchy(chain_idx):
            if chain_idx == len(chain):
                return [""]
                
            parents = build_hierarchy(chain_idx + 1)
            
            current_level = []
            for item_id in chain[chain_idx]['ids']:
                for p in parents:
                    current_level.append({
                        "type": chain[chain_idx]['type'],
                        "index": item_id,
                        "parent_ref": copy.deepcopy(p) if p else ""
                    })
            return current_level
            
        results.extend(build_hierarchy(0))
        # Skip 'i' past all the blocks we just merged or chained
        i = j 
        
    return results

def get_clean_text(paragraph):
    text = ""
    para_size_pt = None
    if paragraph.style and paragraph.style.font and paragraph.style.font.size:
        para_size_pt = paragraph.style.font.size.pt

    for run in paragraph.runs:
        if run.style and run.style.name == 'Footnote Reference':
            continue
        if run.font and run.font.superscript:
            continue
            
        run_size_pt = run.font.size.pt if run.font.size else para_size_pt
        if run_size_pt is not None and run_size_pt <= 12.0:
            continue
            
        text += run.text
        
    text = re.sub(r'\[\d+\]', '', text)
    return text.strip()

def parse_docx_to_json(file_path):
    doc = docx.Document(file_path)
    
    document_data = []
    
    current_chuong = None
    current_dieu = None
    current_khoan = None
    current_diem = None

    chapter_pattern = re.compile(r'^Chương\s*(?:([IVXLCDM]+)(?:\s+|-|:|\.|$))?(.*)', re.IGNORECASE)
    dieu_pattern = re.compile(r'^Điều\s+(\d+[a-zđ]*)\.?\s*(.*)', re.IGNORECASE)
    khoan_pattern = re.compile(r'^(\d+)\.(?:\d+)?\s*(.*)')
    diem_pattern = re.compile(r'^([a-zđ])\)(?:\d+)?\s*(.*)')
    
    footnote_text_pattern = re.compile(r'^\d+\s*(?:Điều này|Khoản này|Điểm này|Cụm từ|Từ|Chữ|Bãi bỏ|Sửa đổi|Bổ sung|Thay thế|Theo quy định).*|^\d+$', re.IGNORECASE)

    # String containing all standard Vietnamese characters (for our negative lookbehind below)
    VN_CHARS = 'a-zA-ZĐđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ'

    for para in doc.paragraphs:
        raw_text = get_clean_text(para)
        
        if not raw_text:
            continue
            
        # FIX 1: Split merged Khoản/Điểm separated by tabs or punctuation+spaces 
        raw_text = re.sub(r'\s*\t\s*(?=\d+\.(?:\d+)?\s|[a-zđ]\)(?:\d+)?\s)', '\n', raw_text)
        raw_text = re.sub(r'(?<=[.;:])\s+(?=\d+\.(?:\d+)?\s|[a-zđ]\)(?:\d+)?\s)', '\n', raw_text)

        # FIX 2: Split Khoản merged directly to the end of an Article title without punctuation
        # Lookbehinds ensure we don't accidentally split "Điều 1. " or "Khoản 1. "
        raw_text = re.sub(rf'(?i)(?<!\bđiều)(?<!\bkhoản)(?<!\bchương)(?<!\bđiểm)(?<=[{VN_CHARS}])\s+(?=\d+\.(?:\d+)?\s)', '\n', raw_text)

        # FIX 3: Weld isolated keywords back to their text if separated by a soft line break
        raw_text = re.sub(r'(?i)(^|\n)(Chương|Điều|Khoản|Điểm)\s*\n\s*', r'\1\2 ', raw_text)
            
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        for text in lines:
            if footnote_text_pattern.match(text):
                continue
                
            chapter_match = chapter_pattern.match(text)
            if chapter_match:
                index = chapter_match.group(1).upper() if chapter_match.group(1) else "Unknown"
                title = chapter_match.group(2).strip(" -.:\t")
                
                current_chuong = {
                    "type": "Chương",
                    "index": index,
                    "text": title,
                    "Điều": []
                }
                document_data.append(current_chuong)
                current_dieu = current_khoan = current_diem = None
                continue

            dieu_match = dieu_pattern.match(text)
            if dieu_match:
                if current_chuong is None:
                    current_chuong = {"type": "Chương", "index": "Unknown", "text": "", "Điều": []}
                    document_data.append(current_chuong)
                    
                current_dieu = {
                    "type": "Điều",
                    "index": dieu_match.group(1),
                    "text": dieu_match.group(2),
                    "Khoản": [],
                    "mentions": extract_mentions(dieu_match.group(2))
                }
                current_chuong["Điều"].append(current_dieu)
                current_khoan = current_diem = None
                continue

            khoan_match = khoan_pattern.match(text)
            if khoan_match and current_dieu is not None:
                current_khoan = {
                    "type": "Khoản",
                    "index": khoan_match.group(1),
                    "text": khoan_match.group(2),
                    "Điểm": [],
                    "mentions": extract_mentions(khoan_match.group(2))
                }
                current_dieu["Khoản"].append(current_khoan)
                current_diem = None
                continue

            diem_match = diem_pattern.match(text)
            if diem_match and current_khoan is not None:
                current_diem = {
                    "type": "Điểm",
                    "index": diem_match.group(1),
                    "text": diem_match.group(2),
                    "mentions": extract_mentions(diem_match.group(2))
                }
                current_khoan["Điểm"].append(current_diem)
                continue

            # Continuation lines
            if current_diem is not None:
                current_diem["text"] += "\n" + text
                current_diem["mentions"].extend(extract_mentions(text))
            elif current_khoan is not None:
                current_khoan["text"] += "\n" + text
                current_khoan["mentions"].extend(extract_mentions(text))
            elif current_dieu is not None:
                current_dieu["text"] += "\n" + text
                current_dieu["mentions"].extend(extract_mentions(text))
            elif current_chuong is not None:
                current_chuong["text"] += "\n" + text

    return document_data

if __name__ == "__main__":
    input_file = r"C:\Users\hungn\Downloads\darkin\2025.docx" 
    output_file = r"C:\Users\hungn\Downloads\darkin\2025_parsed.json"
    
    try:
        parsed_json = parse_docx_to_json(input_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=4)
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")