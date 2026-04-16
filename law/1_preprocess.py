import docx
import re
import json

class Stage1_Preprocessor:
    def __init__(self):
        self.footnote_text_pattern = re.compile(r'^\d+\s*(?:Điều này|Khoản này|Điểm này|Cụm từ|Từ|Chữ|Bãi bỏ|Sửa đổi|Bổ sung|Thay thế|Theo quy định).*|^\d+$', re.IGNORECASE)
        self.VN_CHARS = 'a-zA-ZĐđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ'

    def _get_clean_text(self, paragraph):
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
            
        return re.sub(r'\[\d+\]', '', text).strip()

    def process(self, file_path):
        doc = docx.Document(file_path)
        clean_lines = []

        for para in doc.paragraphs:
            raw_text = self._get_clean_text(para)
            if not raw_text:
                continue
            raw_text = re.sub(r'(?i)\s*\t\s*(?=Điều\s+\d+|Chương\s+[IVXLCDM]+)', '\n', raw_text)
            raw_text = re.sub(r'\s*\t\s*(?=\d{1,2}\.(?!\d)\s|[a-zđ]\)(?:\d+)?\s)', '\n', raw_text)
            raw_text = re.sub(r'(?<=[.;:])\s+(?=\d{1,2}\.(?!\d)\s|[a-zđ]\)(?:\d+)?\s)', '\n', raw_text)
            raw_text = re.sub(rf'(?i)(?<!\bđiều)(?<!\bkhoản)(?<!\bchương)(?<!\bđiểm)(?<=[{self.VN_CHARS}])\s+(?=\d{{1,2}}\.(?!\d)\s)', '\n', raw_text)
            raw_text = re.sub(r'(?i)(^|\n)(Chương|Điều|Khoản|Điểm)\s*\n\s*', r'\1\2 ', raw_text)
            
            lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
            for text in lines:
                if not self.footnote_text_pattern.match(text):
                    clean_lines.append(text)
                    
        return clean_lines

if __name__ == "__main__":
    input_file = r"C:\Users\hungn\Downloads\darkin\2025.docx" 
    output_file = r"C:\Users\hungn\Downloads\darkin\stage1_lines.json"
    
    try:
        print("Stage 1: Ingesting and cleaning Word document...")
        preprocessor = Stage1_Preprocessor()
        clean_lines = preprocessor.process(input_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_lines, f, ensure_ascii=False, indent=4)
            
        print(f"✅ Success! Extracted {len(clean_lines)} lines to {output_file}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")