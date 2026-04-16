"""
transform_case.py
-----------------
Transforms a Vietnamese caselaw input.json into the structured template.json format.

Strategy:
  - Direct Python mapping (no API): Ma_Ban_An, NOI_DUNG_VU_AN
  - Targeted OpenAI calls (gpt-4o with Structured Outputs):
      * THONG_TIN_CHUNG  ← Danh_sach_bi_cao
      * De_Nghi_Cua_Vien_Kiem_Sat  ← Noi_dung_vu_an.Ket_luan_cac_ben
      * NHAN_DINH_CUA_TOA_AN  ← Nhan_dinh_cua_toa_an
      * PHAN_QUYET_CUA_TOA_SO_THAM  ← Phan_quyet

Usage:
    OPENAI_API_KEY=sk-... python transform_case.py input.json output.json
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Model selection & pricing (per 1M tokens, as of 2025)
# ---------------------------------------------------------------------------
#   gpt-4o-mini  : $0.15/M in, $0.60/M out  — cheap, good at structured extraction
#   gpt-4o       : $2.50/M in, $10.00/M out — best quality
#   gpt-4.1-nano : $0.10/M in, $0.40/M out  — cheapest
#   gpt-4.1-mini : $0.40/M in, $1.60/M out  — balanced
MODEL = "gpt-4o-mini"

PRICING = {
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "gpt-4o":       {"input": 2.50,  "output": 10.00},
    "gpt-4.1-nano": {"input": 0.10,  "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40,  "output": 1.60},
}

if MODEL not in PRICING:
    raise ValueError(f"Unknown model '{MODEL}'. Add it to the PRICING dict or pick one of: {list(PRICING)}")

PRICE_INPUT_PER_1M  = PRICING[MODEL]["input"]
PRICE_OUTPUT_PER_1M = PRICING[MODEL]["output"]


@dataclass
class CallRecord:
    name: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens     / 1_000_000 * PRICE_INPUT_PER_1M
            + self.completion_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M
        )


@dataclass
class UsageTracker:
    file_name: str
    calls: List[CallRecord] = field(default_factory=list)

    def record(self, name: str, usage) -> None:
        self.calls.append(CallRecord(
            name=name,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        ))

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def print_report(self) -> None:
        sep = "─" * 72
        print(f"\n{sep}")
        print(f"  TOKEN & COST REPORT  ·  {self.file_name}")
        print(sep)
        header = f"  {'Call':<38} {'In':>8} {'Out':>8} {'Total':>8}  {'Cost (USD)':>10}"
        print(header)
        print(f"  {'-'*38} {'-'*8} {'-'*8} {'-'*8}  {'-'*10}")
        for c in self.calls:
            print(
                f"  {c.name:<38} {c.prompt_tokens:>8,} {c.completion_tokens:>8,} "
                f"{c.total_tokens:>8,}  ${c.cost_usd:>9.5f}"
            )
        print(f"  {'-'*38} {'-'*8} {'-'*8} {'-'*8}  {'-'*10}")
        print(
            f"  {'TOTAL':<38} {self.total_prompt_tokens:>8,} "
            f"{self.total_completion_tokens:>8,} {self.total_tokens:>8,}  "
            f"${self.total_cost_usd:>9.5f}"
        )
        print(sep + "\n")

# ---------------------------------------------------------------------------
# JSON Schemas for Structured Outputs
# ---------------------------------------------------------------------------

SCHEMA_THONG_TIN_CHUNG = {
    "type": "object",
    "properties": {
        "Ngay": {"type": "string", "description": "Ngày tuyên án, định dạng YYYY-MM-DD"},
        "Bi_cao": {
            "type": "string",
            "description": "Giữ nguyên đầy đủ thông tin bị cáo từ văn bản gốc"
        },
        "Bi_hai": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "info": {"type": "string", "description": "Thông tin ngắn gọn về bị hại"}
                },
                "required": ["name", "info"],
                "additionalProperties": False
            }
        },
        "Nguoi_Khac": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"}
                },
                "required": ["name", "role"],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "Ngay", "Bi_cao", "Bi_hai", "Nguoi_Khac"
    ],
    "additionalProperties": False
}

DIEU_LUAT_ITEM = {
    "type": "object",
    "properties": {
        "Diem": {"type": "string"},
        "Khoan": {"type": "string"},
        "Dieu": {"type": "string"},
        "Bo_Luat_Va_Van_Ban_Khac": {"type": "string"}
    },
    "required": ["Diem", "Khoan", "Dieu", "Bo_Luat_Va_Van_Ban_Khac"],
    "additionalProperties": False
}

SCHEMA_DE_NGHI = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "Bi_Cao": {"type": "string"},
            "Pham_Toi": {"type": "string"},
            "Can_Cu_Dieu_Luat": {"type": "array", "items": DIEU_LUAT_ITEM},
            "Phat_Tu": {"type": ["string", "null"]},
            "Phat_Tien": {"type": ["string", "null"]},
            "An_Phi": {"type": ["string", "null"]},
            "Hinh_Phat_Bo_Sung": {"type": ["string", "null"]},
            "Trach_Nhiem_Dan_Su": {"type": ["string", "null"]},
            "Xu_Ly_Vat_Chung": {"type": ["string", "null"]}
        },
        "required": [
            "Bi_Cao", "Pham_Toi", "Can_Cu_Dieu_Luat",
            "Phat_Tu", "Phat_Tien", "An_Phi",
            "Hinh_Phat_Bo_Sung", "Trach_Nhiem_Dan_Su", "Xu_Ly_Vat_Chung"
        ],
        "additionalProperties": False
    }
}

SCHEMA_NHAN_DINH = {
    "type": "object",
    "properties": {
        "Chung_Cu_Xac_Dinh_Co_Toi": {"type": "string"},
        "Chung_Cu_Xac_Dinh_Khong_Co_Toi": {"type": ["string", "null"]},
        "Co_Toi": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Bi_Cao": {"type": "string"},
                    "Toi_Danh": {"type": ["string", "null"]},
                    "Tinh_Tiet_Tang_Nang": {"type": "array", "items": {"type": "string"}},
                    "Tinh_Tiet_Giam_Nhe": {"type": "array", "items": {"type": "string"}},
                    "Huong_Xu_Ly": {"type": "string"}
                },
                "required": [
                    "Bi_Cao", "Toi_Danh", "Tinh_Tiet_Tang_Nang",
                    "Tinh_Tiet_Giam_Nhe", "Huong_Xu_Ly"
                ],
                "additionalProperties": False
            }
        },
        "Phan_Tich": {"type": "string"}
    },
    "required": [
        "Chung_Cu_Xac_Dinh_Co_Toi", "Chung_Cu_Xac_Dinh_Khong_Co_Toi",
        "Co_Toi", "Phan_Tich"
    ],
    "additionalProperties": False
}

SCHEMA_PHAN_QUYET = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "Bi_Cao": {"type": "string"},
            "Can_Cu_Dieu_Luat": {"type": "array", "items": DIEU_LUAT_ITEM},
            "Pham_Toi": {"type": "string"},
            "Phat_Tu": {"type": ["string", "null"]},
            "Phat_Tien": {"type": ["string", "null"]},
            "An_Phi": {"type": ["string", "null"]},
            "Hinh_Phat_Bo_Sung": {"type": ["string", "null"]},
            "Trach_Nhiem_Dan_Su": {"type": ["string", "null"]},
            "Xu_Ly_Vat_Chung": {"type": ["string", "null"]}
        },
        "required": [
            "Bi_Cao", "Can_Cu_Dieu_Luat", "Pham_Toi",
            "Phat_Tu", "Phat_Tien", "An_Phi",
            "Hinh_Phat_Bo_Sung", "Trach_Nhiem_Dan_Su", "Xu_Ly_Vat_Chung"
        ],
        "additionalProperties": False
    }
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_ma_ban_an(filename: str) -> str:
    """
    Extract case code from filename.
    E.g. '13-01-2026-Thai_Nguyen-2ta2045649t1cvn.md' → attempt to pull a code
    like '112/2025/HS-ST' if embedded, otherwise return the stem.
    Filenames don't always contain the full code, so we return the stem as fallback.
    """
    stem = re.sub(r"\.md$", "", filename, flags=re.IGNORECASE)
    return stem


def call_openai(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    schema: dict,
    schema_name: str,
    tracker: UsageTracker,
) -> dict:
    """
    Call OpenAI with Structured Outputs (json_schema response_format).
    Records token usage into tracker. Returns parsed Python object.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema
            }
        },
        max_tokens=4096,
    )
    tracker.record(schema_name, response.usage)
    raw = response.choices[0].message.content
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Main transform function
# ---------------------------------------------------------------------------

def transform(input_path: str, output_path: str) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    model = MODEL

    # Load input
    with open(input_path, "r", encoding="utf-8") as f:
        inp = json.load(f)

    # -----------------------------------------------------------------------
    # 1. DIRECT COPY-PASTE (no API calls)
    # -----------------------------------------------------------------------
    ma_ban_an = extract_ma_ban_an(inp.get("filename", ""))
    noi_dung_vu_an = inp["Noi_dung_vu_an"]["Qua_trinh_dieu_tra"]
    ket_luan_cac_ben = inp["Noi_dung_vu_an"]["Ket_luan_cac_ben"]

    tracker = UsageTracker(file_name=os.path.basename(input_path))

    # -----------------------------------------------------------------------
    # 2. TARGETED AI CALLS
    # -----------------------------------------------------------------------

    # --- 2a. THONG_TIN_CHUNG (party information) ---
    print("[1/4] Extracting THONG_TIN_CHUNG...")
    thong_tin_system = (
        "Bạn là trợ lý pháp lý chuyên trích xuất thông tin từ bản án hình sự Việt Nam. "
        "Trả về JSON theo đúng schema được yêu cầu. "
        "Trích xuất chính xác từ văn bản đầu vào, không suy đoán thông tin không có trong văn bản. "
        "Giữ nguyên đầy đủ phần bị cáo vào trường Bi_cao. "
        "Bi_hai chỉ tóm tắt ngắn gọn mỗi người. "
        "Các vai trò khác gom vào Nguoi_Khac với từng phần tử gồm name và role. "
        "Nếu không tìm thấy thông tin, dùng chuỗi rỗng '' cho string và mảng rỗng [] cho array."
    )
    thong_tin_user = (
        "Trích xuất thông tin từ phần giới thiệu các bên tham gia tố tụng trong bản án dưới đây.\n\n"
        "Văn bản nhân thân bị cáo:\n"
        + "\n---\n".join(inp.get("Danh_sach_bi_cao", []))
        + "\n\nVăn bản kết luận các bên (để trích xuất người bào chữa, người bảo vệ, bị hại, đương sự, người tham gia tố tụng khác và ngày tuyên án):\n"
        + ket_luan_cac_ben
    )
    thong_tin_data = call_openai(
        client, model, thong_tin_system, thong_tin_user,
        SCHEMA_THONG_TIN_CHUNG, "ThongTinChung", tracker
    )
    print(f"    → {tracker.calls[-1].total_tokens:,} tokens  (${tracker.calls[-1].cost_usd:.5f})")

    # --- 2b. De_Nghi_Cua_Vien_Kiem_Sat ---
    print("[2/4] Extracting De_Nghi_Cua_Vien_Kiem_Sat...")
    de_nghi_system = (
        "Bạn là trợ lý pháp lý chuyên trích xuất thông tin từ bản án hình sự Việt Nam. "
        "Trích xuất ĐỀ NGHỊ CỦA VIỆN KIỂM SÁT (không phải phán quyết của Tòa án). "
        "Đây là phần luận tội, đề nghị mức hình phạt của Kiểm sát viên tại phiên tòa. "
        "Nếu không có thông tin, dùng null."
    )
    de_nghi_user = (
        "Từ đoạn văn bản sau (phần kết luận các bên), "
        "trích xuất đề nghị của Viện kiểm sát (VKS) đối với từng bị cáo:\n\n"
        + ket_luan_cac_ben
    )
    # Wrap array schema in an object for strict mode compatibility
    de_nghi_wrapper_schema = {
        "type": "object",
        "properties": {
            "items": SCHEMA_DE_NGHI
        },
        "required": ["items"],
        "additionalProperties": False
    }
    de_nghi_raw = call_openai(
        client, model, de_nghi_system, de_nghi_user,
        de_nghi_wrapper_schema, "DeNghiCuaVienKiemSat", tracker
    )
    de_nghi_data = de_nghi_raw["items"]
    print(f"    → {tracker.calls[-1].total_tokens:,} tokens  (${tracker.calls[-1].cost_usd:.5f})")

    # --- 2c. NHAN_DINH_CUA_TOA_AN ---
    print("[3/4] Extracting NHAN_DINH_CUA_TOA_AN...")
    nhan_dinh_system = (
        "Bạn là trợ lý pháp lý chuyên trích xuất thông tin từ bản án hình sự Việt Nam. "
        "Trích xuất NHẬN ĐỊNH CỦA TÒA ÁN từ các phần nhận định được đánh số [1], [2], [3]... "
        "Phần [1] thường về tính hợp pháp tố tụng. "
        "Phần [2], [3]... thường về chứng cứ, tội danh, hình phạt. "
        "Nếu không có thông tin, dùng null hoặc mảng rỗng."
    )
    nhan_dinh_input = json.dumps(inp.get("Nhan_dinh_cua_toa_an", {}), ensure_ascii=False, indent=2)
    nhan_dinh_user = (
        "Từ các phần nhận định của Hội đồng xét xử sau, "
        "trích xuất toàn bộ thông tin theo schema:\n\n"
        + nhan_dinh_input
    )
    nhan_dinh_data = call_openai(
        client, model, nhan_dinh_system, nhan_dinh_user,
        SCHEMA_NHAN_DINH, "NhanDinhCuaToaAn", tracker
    )
    print(f"    → {tracker.calls[-1].total_tokens:,} tokens  (${tracker.calls[-1].cost_usd:.5f})")

    # --- 2d. PHAN_QUYET_CUA_TOA_SO_THAM ---
    print("[4/4] Extracting PHAN_QUYET_CUA_TOA_SO_THAM...")
    phan_quyet_system = (
        "Bạn là trợ lý pháp lý chuyên trích xuất thông tin từ bản án hình sự Việt Nam. "
        "Trích xuất PHÁN QUYẾT CHÍNH THỨC CỦA TÒA ÁN từ phần QUYẾT ĐỊNH. "
        "ĐÂY KHÔNG PHẢI đề nghị của Viện kiểm sát. "
        "Lấy từ phần 'raw_text' và 'Can_cu_dieu_luat', 'Pham_toi', 'Phat_tu'... trong dữ liệu đầu vào. "
        "Nếu có nhiều bị cáo, trích xuất riêng cho từng người."
    )
    phan_quyet_input = json.dumps(inp.get("Phan_quyet", {}), ensure_ascii=False, indent=2)
    phan_quyet_user = (
        "Từ phần quyết định của bản án sau, "
        "trích xuất phán quyết chính thức của Tòa án cho từng bị cáo:\n\n"
        + phan_quyet_input
    )
    phan_quyet_wrapper_schema = {
        "type": "object",
        "properties": {
            "items": SCHEMA_PHAN_QUYET
        },
        "required": ["items"],
        "additionalProperties": False
    }
    phan_quyet_raw = call_openai(
        client, model, phan_quyet_system, phan_quyet_user,
        phan_quyet_wrapper_schema, "PhanQuyetCuaToaSoTham", tracker
    )
    phan_quyet_data = phan_quyet_raw["items"]
    print(f"    → {tracker.calls[-1].total_tokens:,} tokens  (${tracker.calls[-1].cost_usd:.5f})")

    # -----------------------------------------------------------------------
    # 3. ASSEMBLE OUTPUT
    # -----------------------------------------------------------------------
    output = {
        "THONG_TIN_CHUNG": {
            "Ma_Ban_An": ma_ban_an,
            "Ngay": thong_tin_data.get("Ngay", ""),
            "Bi_cao": thong_tin_data.get("Bi_cao", ""),
            "Bi_hai": thong_tin_data.get("Bi_hai", []),
            "Nguoi_Khac": thong_tin_data.get("Nguoi_Khac", [])
        },
        "NOI_DUNG_VU_AN": noi_dung_vu_an,
        "De_Nghi_Cua_Vien_Kiem_Sat": de_nghi_data,
        "NHAN_DINH_CUA_TOA_AN": nhan_dinh_data,
        "PHAN_QUYET_CUA_TOA_SO_THAM": phan_quyet_data,
        "_usage": {
            "model": model,
            "pricing": {
                "input_per_1m_usd": PRICE_INPUT_PER_1M,
                "output_per_1m_usd": PRICE_OUTPUT_PER_1M,
            },
            "calls": [
                {
                    "name": c.name,
                    "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens,
                    "total_tokens": c.total_tokens,
                    "cost_usd": round(c.cost_usd, 6),
                }
                for c in tracker.calls
            ],
            "totals": {
                "prompt_tokens": tracker.total_prompt_tokens,
                "completion_tokens": tracker.total_completion_tokens,
                "total_tokens": tracker.total_tokens,
                "cost_usd": round(tracker.total_cost_usd, 6),
            },
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    tracker.print_report()
    print(f"✅ Output written to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transform Vietnamese caselaw input.json → structured output.json"
    )
    parser.add_argument("input",  help="Path to input JSON file")
    parser.add_argument("output", help="Path for output JSON file")
    parser.add_argument(
        "--model",
        default=MODEL,
        choices=list(PRICING),
        help=f"OpenAI model to use (default: {MODEL}). "
             "Choices: " + ", ".join(
                 f"{m} (${p['input']}/M in, ${p['output']}/M out)"
                 for m, p in PRICING.items()
             ),
    )
    args = parser.parse_args()

    # Allow CLI override of model and pricing
    if args.model != MODEL:
        MODEL = args.model  # type: ignore[assignment]
        PRICE_INPUT_PER_1M  = PRICING[MODEL]["input"]   # type: ignore[assignment]
        PRICE_OUTPUT_PER_1M = PRICING[MODEL]["output"]  # type: ignore[assignment]

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    print(f"Model : {MODEL}  (${PRICE_INPUT_PER_1M}/M in · ${PRICE_OUTPUT_PER_1M}/M out)")
    print(f"Input : {args.input}")
    print(f"Output: {args.output}\n")

    transform(args.input, args.output)