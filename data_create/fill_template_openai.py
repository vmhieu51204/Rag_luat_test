import json
import os
import glob
import argparse
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Lazily initialize OpenAI client to allow CLI parsing without API key."""
    global client
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return client

# ==========================================
# 0. PRICING CONFIGURATION
# ==========================================
MODEL_NAME = "gpt-5.4-nano"
INPUT_PRICE_PER_1M_USD = 0.15
OUTPUT_PRICE_PER_1M_USD = 0.60

# ==========================================
# 1. DEFINE PYDANTIC SCHEMAS FOR LLM EXTRACTION
# ==========================================
class CanCuDieuLuat(BaseModel):
    Diem: Optional[str] = Field(None, description="Legal point (điểm)")
    Khoan: Optional[str] = Field(None, description="Legal clause (khoản)")
    Dieu: Optional[str] = Field(None, description="Legal article (Điều)")
    Bo_Luat_Va_Van_Ban_Khac: Optional[str] = Field(None, description="Law or other document (Bộ luật)")

class ThongTinBiCao(BaseModel):
    Ho_Ten: str
    Ngay_Sinh: Optional[str] = None
    Noi_Cu_Tru: Optional[str] = None
    Nghe_Nghiep: Optional[str] = None
    Trinh_Do_Van_Hoa: Optional[str] = None
    Dan_Toc: Optional[str] = None
    Gioi_Tinh: Optional[str] = None
    Ton_Giao: Optional[str] = None
    Quoc_Tich: Optional[str] = None
    Hoan_Canh_Gia_Dinh: Optional[str] = None
    Tien_An: Optional[str] = None
    Tien_Su: Optional[str] = None
    Chi_Tiet_Nhan_Than: Optional[str] = None
    Ngay_Tam_Giam: Optional[str] = None
    Trang_Thai_Co_Mat: Optional[str] = None



class DeNghiVienKiemSat(BaseModel):
    Bi_Cao: str
    Pham_Toi: str
    Can_Cu_Dieu_Luat: List[CanCuDieuLuat]
    Phat_Tu: Optional[str] = None
    Phat_Tien: Optional[str] = None
    An_Phi: Optional[str] = None
    Hinh_Phat_Bo_Sung: Optional[str] = None
    Trach_Nhiem_Dan_Su: Optional[str] = None
    Xu_Ly_Vat_Chung: Optional[str] = None

class PhanQuyetToaSoTham(BaseModel):
    Bi_Cao: str
    Can_Cu_Dieu_Luat: List[CanCuDieuLuat]
    Pham_Toi: str
    Phat_Tu: Optional[str] = None
    Phat_Tien: Optional[str] = None
    An_Phi: Optional[str] = None
    Hinh_Phat_Bo_Sung: Optional[str] = None
    Trach_Nhiem_Dan_Su: Optional[str] = None
    Xu_Ly_Vat_Chung: Optional[str] = None

class LLMExtractionOutput(BaseModel):
    Thong_Tin_Bi_Cao: List[ThongTinBiCao]
    De_Nghi_Cua_Vien_Kiem_Sat: List[DeNghiVienKiemSat]
    PHAN_QUYET_CUA_TOA_SO_THAM: List[PhanQuyetToaSoTham]


class VerdictOnlyOutput(BaseModel):
    PHAN_QUYET_CUA_TOA_SO_THAM: List[PhanQuyetToaSoTham]

# ==========================================
# 2. CORE PROCESSING LOGIC
# ==========================================
def process_caselaw_file(input_filepath: str, output_filepath: str) -> float:
    """
    Processes a single stage2 JSON file, extracts data using LLM, saves to stage3,
    and returns the cost in USD.
    """
    with open(input_filepath, 'r', encoding='utf-8') as f:
        stage2_data = json.load(f)
        
    filename = os.path.basename(input_filepath)
    file_cost = 0.0
    openai_client = get_client()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    usage_calls = []

    def add_usage_call(call_name: str, usage):
        nonlocal file_cost, total_prompt_tokens, total_completion_tokens, total_tokens, usage_calls
        prompt_cost = (usage.prompt_tokens / 1_000_000) * INPUT_PRICE_PER_1M_USD
        completion_cost = (usage.completion_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M_USD
        call_cost = prompt_cost + completion_cost
        file_cost += call_cost

        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        total_tokens += usage.total_tokens

        usage_calls.append(
            {
                "name": call_name,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": round(call_cost, 6),
            }
        )
    
    # 1. COPY PASTE FIELDS 
    stage3_data = {
        "THONG_TIN_CHUNG": {
            "Ma_Ban_An": filename.replace(".json", ""),
            "Thong_Tin_Bi_Cao": [], 
            "Thong_Tin_Nguoi_Tham_Gia_To_Tung": stage2_data.get("Nguoi_tham_gia", "")
        },
        "NOI_DUNG_VU_AN": stage2_data.get("Noi_dung_vu_an", {}).get("Qua_trinh_dieu_tra", ""),
        "De_Nghi_Cua_Vien_Kiem_Sat": [],
        "NHAN_DINH_CUA_TOA_AN": stage2_data.get("Nhan_dinh_cua_toa_an", {}),
        "PHAN_QUYET_CUA_TOA_SO_THAM": []
    }

    llm_input_context = f"""
    --- DANH SÁCH BỊ CÁO VÀ NGƯỜI LIÊN QUAN ---
    {json.dumps(stage2_data.get('Danh_sach_bi_cao', []), ensure_ascii=False)}
    
    --- KẾT LUẬN CỦA CÁC BÊN (VIỆN KIỂM SÁT) ---
    {stage2_data.get('Noi_dung_vu_an', {}).get('Ket_luan_cac_ben', '')}
    
    --- QUYẾT ĐỊNH CỦA TÒA ÁN ---
    {stage2_data.get('QUYET_DINH', '')}
    """

    system_prompt = """
    You are an expert legal data extraction AI. Extract the provided Vietnamese caselaw text into the exact JSON schema requested.
    
    CRITICAL INSTRUCTION FOR LEGAL CITATIONS (Can_Cu_Dieu_Luat):
    You must handle hierarchical legal citations like a "valid parenthesis" distribution problem. Higher-level units (Điều) apply to preceding lower-level units (Khoản, Điểm) in the current phrase. 
    
    Rules for distribution:
    1. "điểm s khoản 1, 2 Điều 51" -> (Dieu 51, Khoan 1, Diem s) AND (Dieu 51, Khoan 2, Diem null). 
       Do NOT apply 'điểm s' to 'khoản 2'.
    2. "điểm c, d khoản 2, khoản 5 Điều 355" -> (Dieu 355, Khoan 2, Diem c) AND (Dieu 355, Khoan 2, Diem d) AND (Dieu 355, Khoan 5, Diem null).
    """

    try:
        response = openai_client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": llm_input_context}
            ],
            response_format=LLMExtractionOutput,
            temperature=0.0
        )
        
        extracted_data = response.choices[0].message.parsed
        add_usage_call("llm_extraction", response.usage)
        
        # Merge LLM output
        stage3_data["THONG_TIN_CHUNG"]["Thong_Tin_Bi_Cao"] = [
            d.model_dump(exclude_none=True) for d in extracted_data.Thong_Tin_Bi_Cao
        ]
        stage3_data["De_Nghi_Cua_Vien_Kiem_Sat"] = [
            p.model_dump(exclude_none=True) for p in extracted_data.De_Nghi_Cua_Vien_Kiem_Sat
        ]
        stage3_data["PHAN_QUYET_CUA_TOA_SO_THAM"] = [
            v.model_dump(exclude_none=True) for v in extracted_data.PHAN_QUYET_CUA_TOA_SO_THAM
        ]

        # Retry focused extraction when decision text exists but verdict list is empty.
        if stage2_data.get("QUYET_DINH", "").strip() and not stage3_data["PHAN_QUYET_CUA_TOA_SO_THAM"]:
            retry_input = f"""
            --- QUYẾT ĐỊNH CỦA TÒA ÁN ---
            {stage2_data.get('QUYET_DINH', '')}
            """
            retry_system_prompt = """
            You are an expert legal extractor.
            Extract ONLY PHAN_QUYET_CUA_TOA_SO_THAM from the decision text.
            Return at least one item when the text contains any declaration of guilt, sentence, or legal basis.
            Keep legal citations in Can_Cu_Dieu_Luat with fields Diem, Khoan, Dieu, Bo_Luat_Va_Van_Ban_Khac.
            """
            try:
                retry_response = openai_client.beta.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": retry_system_prompt},
                        {"role": "user", "content": retry_input},
                    ],
                    response_format=VerdictOnlyOutput,
                    temperature=0.0,
                )
                retry_data = retry_response.choices[0].message.parsed
                add_usage_call("llm_extraction_verdict_retry", retry_response.usage)
                retry_verdict = [
                    v.model_dump(exclude_none=True) for v in retry_data.PHAN_QUYET_CUA_TOA_SO_THAM
                ]
                if retry_verdict:
                    stage3_data["PHAN_QUYET_CUA_TOA_SO_THAM"] = retry_verdict
                else:
                    stage3_data.setdefault("_warnings", []).append(
                        "PHAN_QUYET_CUA_TOA_SO_THAM is empty after retry"
                    )
            except Exception as retry_error:
                print(f"Retry extraction failed for {filename}: {retry_error}")
                stage3_data.setdefault("_warnings", []).append(
                    "Retry extraction for PHAN_QUYET_CUA_TOA_SO_THAM failed"
                )
        
        # Append Usage Block
        stage3_data["_usage"] = {
            "model": MODEL_NAME,
            "pricing": {
                "input_per_1m_usd": INPUT_PRICE_PER_1M_USD,
                "output_per_1m_usd": OUTPUT_PRICE_PER_1M_USD
            },
            "calls": usage_calls,
            "totals": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(file_cost, 6)
            }
        }

    except Exception as e:
        print(f"Error extracting {filename}: {e}")

    # Save Stage 3 JSON
    os.makedirs(os.path.dirname(output_filepath) or '.', exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(stage3_data, f, ensure_ascii=False, indent=2)
        
    print(f"Processed: {filename} | Cost: ${file_cost:.6f}")
    return file_cost

# ==========================================
# 3. BATCH PROCESSING & AGGREGATION
# ==========================================
def process_directory(input_dir: str, output_dir: str):
    """
    Finds all JSON files in the input directory, processes them,
    and calculates the total running cost.
    """
    total_pipeline_cost = 0.0
    input_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    print(f"Found {len(input_files)} files to process.")
    print("-" * 40)
    
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename.replace("stage2", "stage3")) 
        
        cost = process_caselaw_file(input_file, output_file)
        total_pipeline_cost += cost
        
    print("-" * 40)
    print(f"BATCH COMPLETE")
    print(f"Total Files Processed: {len(input_files)}")
    print(f"FINAL TOTAL COST: ${total_pipeline_cost:.4f}")


def process_selected_files(input_dir: str, output_dir: str, selected_filenames: List[str]):
    """Process only selected filenames from input_dir."""
    total_pipeline_cost = 0.0
    existing_files = []
    missing_files = []

    for filename in selected_filenames:
        input_path = os.path.join(input_dir, filename)
        if os.path.isfile(input_path):
            existing_files.append(input_path)
        else:
            missing_files.append(filename)

    print(f"Selected {len(selected_filenames)} files.")
    print(f"Found {len(existing_files)} files to process.")
    if missing_files:
        print(f"Missing {len(missing_files)} files: {', '.join(missing_files)}")
    print("-" * 40)

    for input_file in existing_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename.replace("stage2", "stage3"))
        cost = process_caselaw_file(input_file, output_file)
        total_pipeline_cost += cost

    print("-" * 40)
    print("BATCH COMPLETE")
    print(f"Total Files Processed: {len(existing_files)}")
    print(f"FINAL TOTAL COST: ${total_pipeline_cost:.4f}")


def parse_filename_list(value: str) -> List[str]:
    """Parse comma-separated filenames into a cleaned list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def read_filename_list_file(list_file_path: str) -> List[str]:
    """Read newline-separated filenames from a text file."""
    with open(list_file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fill stage3 template from stage2 extracted fields."
    )
    parser.add_argument(
        "--input-dir",
        default="extracted_fields",
        help="Directory containing input JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="filled_template_openai",
        help="Directory to write output JSON files",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=None,
        help="Process the first N files (sorted by filename) from input-dir",
    )
    parser.add_argument(
        "--file-list",
        default=None,
        help="Comma-separated list of filenames to process",
    )
    parser.add_argument(
        "--file-list-path",
        default=None,
        help="Path to a txt file containing one filename per line",
    )

    args = parser.parse_args()

    list_mode_enabled = bool(args.file_list or args.file_list_path)
    first_n_mode_enabled = args.first_n is not None

    if list_mode_enabled and first_n_mode_enabled:
        raise ValueError("Use either --first-n or file list mode, not both.")

    if list_mode_enabled:
        selected_filenames: List[str] = []
        if args.file_list:
            selected_filenames.extend(parse_filename_list(args.file_list))
        if args.file_list_path:
            selected_filenames.extend(read_filename_list_file(args.file_list_path))

        # Keep order while removing duplicates.
        deduped = []
        seen = set()
        for name in selected_filenames:
            if name not in seen:
                seen.add(name)
                deduped.append(name)

        if not deduped:
            raise ValueError("No filenames were provided in list mode.")

        process_selected_files(args.input_dir, args.output_dir, deduped)
    elif first_n_mode_enabled:
        if args.first_n < 1:
            raise ValueError("--first-n must be >= 1")

        all_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
        if not all_files:
            raise ValueError(f"No JSON files found in {args.input_dir}")

        selected = [os.path.basename(path) for path in all_files[: args.first_n]]
        process_selected_files(args.input_dir, args.output_dir, selected)
    else:
        process_directory(args.input_dir, args.output_dir)