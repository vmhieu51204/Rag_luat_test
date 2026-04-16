import docx
import re
import json

def extract_mentions(text):
    """
    Extracts mentions of Điều, Khoản, Điểm.
    Fixed regex to avoid matching words like "Điều kiện".
    - Điều/Khoản: expects numbers (e.g., 1, 15, 29a).
    - Điểm: expects a single character (e.g., a, b, đ).
    """
    mentions = []
    
    # Matches: (Điều|Khoản) + space + (numbers optionally followed by letters)
    # OR: (Điểm) + space + (single letter)
    # \b ensures word boundaries so it doesn't match halfway into a word
    pattern = r'\b(Điều|Khoản)\s+([0-9]+[a-zđ]*)\b|\b(Điểm)\s+([a-zđ])\b'
    
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    for match in matches:
        # Since we use OR in regex, either Group 1 & 2 are populated, or Group 3 & 4
        if match.group(1): 
            m_type = match.group(1).capitalize()
            m_index = match.group(2)
        else:
            m_type = match.group(3).capitalize()
            m_index = match.group(4)
        
        mentions.append({
            "type": m_type,
            "index": m_index,
            "parent_ref": ""
        })
        
    return mentions

def parse_docx_to_json(file_path):
    doc = docx.Document(file_path)
    
    document_data = [] # Root list of Chương objects
    
    current_chuong = None
    current_dieu = None
    current_khoan = None
    current_diem = None

    # Regex patterns for identifying structures
    chapter_pattern = re.compile(r'^Chương\s+([IVXLCDM]+)(.*)', re.IGNORECASE)
    dieu_pattern = re.compile(r'^Điều\s+(\d+[a-zđ]*)\.\s*(.*)', re.IGNORECASE)
    khoan_pattern = re.compile(r'^(\d+)\.\s+(.*)')
    diem_pattern = re.compile(r'^([a-zđ])\)\s+(.*)')

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
            
        # 1. Check for Chapter (Chương)
        chapter_match = chapter_pattern.match(text)
        if chapter_match:
            index = chapter_match.group(1).upper()
            title = chapter_match.group(2).strip()
            
            current_chuong = {
                "type": "Chương",
                "index": index,
                "text": title,
                "Điều": []
            }
            document_data.append(current_chuong)
            
            # Reset active children
            current_dieu = None
            current_khoan = None
            current_diem = None
            continue

        # 2. Check for Article (Điều)
        dieu_match = dieu_pattern.match(text)
        if dieu_match:
            # If a Điều appears before any Chương, create a dummy Chương to hold it
            if current_chuong is None:
                current_chuong = {"type": "Chương", "index": "Unknown", "text": "", "Điều": []}
                document_data.append(current_chuong)
                
            index = dieu_match.group(1)
            content = dieu_match.group(2)
            
            current_dieu = {
                "type": "Điều",
                "index": index,
                "text": content,
                "Khoản": [],
                "mentions": extract_mentions(content)
            }
            current_chuong["Điều"].append(current_dieu)
            
            current_khoan = None 
            current_diem = None
            continue

        # 3. Check for Clause (Khoản)
        khoan_match = khoan_pattern.match(text)
        if khoan_match and current_dieu is not None:
            index = khoan_match.group(1)
            content = khoan_match.group(2)
            
            current_khoan = {
                "type": "Khoản",
                "index": index,
                "text": content,
                "Điểm": [],
                "mentions": extract_mentions(content)
            }
            current_dieu["Khoản"].append(current_khoan)
            
            current_diem = None
            continue

        # 4. Check for Point (Điểm)
        diem_match = diem_pattern.match(text)
        if diem_match and current_khoan is not None:
            index = diem_match.group(1)
            content = diem_match.group(2)
            
            current_diem = {
                "type": "Điểm",
                "index": index,
                "text": content,
                "mentions": extract_mentions(content)
            }
            current_khoan["Điểm"].append(current_diem)
            continue

        # 5. Handle continuation lines (multiline text)
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
    # Ensure this matches the correct path to your file
    input_file = r"C:\Users\hungn\Downloads\darkin\2025.docx" 
    
    # You can also specify where you want the JSON to be saved
    output_file = r"C:\Users\hungn\Downloads\darkin\2025.json"
    
    try:
        parsed_json = parse_docx_to_json(input_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=4)
            
        print(f"Successfully converted {input_file} to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")