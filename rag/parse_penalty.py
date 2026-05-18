import re
from typing import List, Optional

def parse_penalty_to_months(text: str) -> Optional[List[int]]:
    """
    Parses a penalty string from Vietnamese (e.g., '08 năm đến 10 năm tù') 
    into a time range in months (e.g., [96, 120]).
    
    Returns:
        A list of two integers representing [min_months, max_months].
        Special values:
            [-1, -1] for "Tù chung thân" (Life imprisonment)
            [-2, -2] for "Tử hình" (Death penalty)
        Returns None if no duration could be parsed.
    """
    if not text:
        return None
    
    text = text.lower()
    
    # Remove content within parentheses to avoid confusing numbers like 12 (mười hai)
    text = re.sub(r'\(.*?\)', '', text)
    
    # If there are multiple penalties but "tổng hợp" is present, usually the final total is what matters
    if "tổng hợp" in text:
        text = text[text.rfind("tổng hợp"):]
        
    # Check for special penalties
    if "tử hình" in text:
        return [-2, -2]
    if "chung thân" in text:
        return [-1, -1]
        
    # Fix missing units in ranges (e.g., '12 đến 13 năm' -> '12 năm đến 13 năm')
    # This handles "đến", "-", and "tới"
    text = re.sub(r'(\d+)\s*(?:đến|-|tới)\s*(\d+)\s*(năm|tháng)', r'\1 \3 đến \2 \3', text)
    
    # Find all occurrences of durations: "X năm Y tháng", "X năm", or "X tháng"
    matches = re.finditer(r'(\d+)\s*năm\s*(\d+)\s*tháng|(\d+)\s*năm|(\d+)\s*tháng', text)
    
    durations = []
    for m in matches:
        if m.group(1) and m.group(2): # X năm Y tháng
            durations.append(int(m.group(1)) * 12 + int(m.group(2)))
        elif m.group(3): # X năm
            durations.append(int(m.group(3)) * 12)
        elif m.group(4): # X tháng
            durations.append(int(m.group(4)))
            
    if not durations:
        return None
    
    # If only one duration found, it's a fixed sentence (min == max)
    if len(durations) == 1:
        return [durations[0], durations[0]]
    # If multiple durations found, take the minimum and maximum to form the range
    elif len(durations) >= 2:
        return [min(durations), max(durations)]
    
    return None

def compute_range_overlap(range1: Optional[List[int]], range2: Optional[List[int]]) -> float:
    """
    Computes the Intersection over Union (IoU) of two time ranges in months.
    Returns a float between 0.0 and 1.0.
    """
    if not range1 or not range2:
        return 0.0
    
    min1, max1 = range1
    min2, max2 = range2
    
    if min1 == min2 and min1 < 0:
        return 1.0
    if min1 < 0 or min2 < 0:
        return 0.0
        
    intersect_min = max(min1, min2)
    intersect_max = min(max1, max2)
    intersection = max(0, intersect_max - intersect_min)
    
    union_min = min(min1, min2)
    union_max = max(max1, max2)
    union = max(0, union_max - union_min)
    
    if union == 0:
        return 1.0 if intersection == 0 and min1 == min2 else 0.0
        
    return intersection / union

if __name__ == "__main__":
    examples = [
        "Phạt bị cáo từ 01 năm đến 01 năm 06 tháng tù.",
        "Từ 02 năm đến 02 năm 06 tháng tù nhưng cho hưởng án treo",
        "12 (mười hai) đến 13 (mười ba) năm tù",
        "từ 19 năm đến 22 năm tù",
        "4 - 5 năm tù",
        "từ 24 đến 36 tháng tù",
        "13-15 năm tù",
        "từ 17 năm 06 tháng đến 18 năm tù",
        "01 (Một) - 02 (H4) năm tù về tội A và 08 (T3) - 09 (C) năm tù về tội B; tổng hợp hình phạt chung 09 - 11 năm",
        "từ 10 năm 06 tháng tù đến 11 năm 06 tháng tù",
        "Tù chung thân",
        "Tử hình",
        "3 to 4 years imprisonment", # This script expects VN terms, so it returns None
        "08 năm đến 10 năm tù",
        "1 năm 6 tháng"
    ]

    for ex in examples:
        print(f"{ex[:45]:<45} -> {parse_penalty_to_months(ex)}")
