import re
import json
import copy

class Stage3_Resolver:
    def __init__(self):
        self.pattern = r'\b(?:các\s+)?(Chương|Điều|Khoản|Điểm)\s+((?:(?:\b[IVXLCDM]+\b|\b[0-9]+[a-zđ]*\b|\b[a-zđ]\b)(?:\s*,\s*|\s+và\s+|\s+hoặc\s+|\s+-\s+)*)+)'
        self.rank = {"Điểm": 1, "Khoản": 2, "Điều": 3, "Chương": 4}

    def _extract_mentions(self, text):
        matches = list(re.finditer(self.pattern, text, re.IGNORECASE))
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
                
            raw_blocks.append({"type": m_type, "ids": ids, "start": match.start(), "end": match.end()})
            
        results = []
        i = 0
        while i < len(raw_blocks):
            chain = [{"type": raw_blocks[i]['type'], "ids": list(raw_blocks[i]['ids']), "start": raw_blocks[i]['start'], "end": raw_blocks[i]['end']}]
            j = i + 1
            
            while j < len(raw_blocks):
                gap_text = text[chain[-1]['end']:raw_blocks[j]['start']]
                gap_valid = re.fullmatch(r'\s*(của|thuộc|tại|,|và|hoặc|-)?\s*', gap_text, re.IGNORECASE)
                
                if not gap_valid:
                    break
                    
                current_rank = self.rank.get(chain[-1]['type'], 0)
                next_rank = self.rank.get(raw_blocks[j]['type'], 0)
                
                if next_rank == current_rank:
                    chain[-1]['ids'].extend(raw_blocks[j]['ids'])
                    chain[-1]['end'] = raw_blocks[j]['end']
                    j += 1
                elif next_rank > current_rank:
                    chain.append({"type": raw_blocks[j]['type'], "ids": list(raw_blocks[j]['ids']), "start": raw_blocks[j]['start'], "end": raw_blocks[j]['end']})
                    j += 1
                else:
                    break
                    
            def build_hierarchy(chain_idx):
                if chain_idx == len(chain): return [""]
                parents = build_hierarchy(chain_idx + 1)
                current_level = []
                for item_id in chain[chain_idx]['ids']:
                    for p in parents:
                        current_level.append({"type": chain[chain_idx]['type'], "index": item_id, "parent_ref": copy.deepcopy(p) if p else ""})
                return current_level
                
            results.extend(build_hierarchy(0))
            i = j 
            
        return results

    def process(self, document_data):
        def traverse_and_resolve(node):
            if isinstance(node, list):
                for item in node:
                    traverse_and_resolve(item)
            elif isinstance(node, dict):
                if "text" in node:
                    node["mentions"] = self._extract_mentions(node["text"])
                for key in ["Điều", "Khoản", "Điểm"]:
                    if key in node:
                        traverse_and_resolve(node[key])
                        
        traverse_and_resolve(document_data)
        return document_data

if __name__ == "__main__":
    input_file = r"C:\Users\hungn\Downloads\darkin\stage2_structure.json" 
    output_file = r"C:\Users\hungn\Downloads\darkin\stage3_final.json"
    
    try:
        print("Stage 3: Resolving cross-references and injecting mentions...")
        with open(input_file, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
            
        resolver = Stage3_Resolver()
        final_data = resolver.process(structured_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
            
        print(f"✅ Success! Saved final JSON to {output_file}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")