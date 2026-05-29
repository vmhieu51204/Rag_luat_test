"""Shared Pydantic schemas and prompt-schema builder for data_create scripts.

This module centralizes the Pydantic models used by the `fill_template_*` scripts
and exposes a helper that renders a prompt-friendly schema including per-field
instructions (copied from Field.description). The rag/generation helper is
reused to keep formatting consistent with the rest of the repo.
"""

from __future__ import annotations

from typing import Any, get_args, get_origin, List, Optional
import json

from pydantic import BaseModel, Field

from rag.generation.schemas import build_output_schema_instruction


class CanCuDieuLuat(BaseModel):
    Diem: Optional[str] = Field(None, description="Legal point (điểm). Provide lowercase letter like 'a' or null if not present.")
    Khoan: Optional[str] = Field(None, description="Legal clause (khoản) number. Use numeric string like '1' or null if not present.")
    Dieu: Optional[str] = Field(None, description="Legal article (Điều) number. Use numeric string like '173' or null if not present.")
    Bo_Luat_Va_Van_Ban_Khac: Optional[str] = Field(None, description="Law or other legal document identifier (e.g., BLHS). Keep the short code or full name if known.")


class ThongTinBiCao(BaseModel):
    Ho_Ten: str = Field(..., description="Defendant full name exactly as it appears in the case text.")
    Ngay_Sinh: Optional[str] = Field(None, description="Date of birth when available. Prefer ISO-ish formats 'YYYY-MM-DD' or a concise representation from text.")
    Noi_Cu_Tru: Optional[str] = Field(None, description="Place of residence string extracted from the text; keep as concise address or locality.")
    Nghe_Nghiep: Optional[str] = Field(None, description="Occupation if mentioned; otherwise null.")
    Trinh_Do_Van_Hoa: Optional[str] = Field(None, description="Education level if mentioned; otherwise null.")
    Dan_Toc: Optional[str] = Field(None, description="Ethnicity when present; otherwise null.")
    Gioi_Tinh: Optional[str] = Field(None, description="Gender if determinable (Nam/Nữ) or null.")
    Ton_Giao: Optional[str] = Field(None, description="Religion if mentioned; otherwise null.")
    Quoc_Tich: Optional[str] = Field(None, description="Nationality if mentioned; otherwise null.")
    Hoan_Canh_Gia_Dinh: Optional[str] = Field(None, description="Concise family circumstance summary if present; otherwise null.")
    Tien_An: Optional[str] = Field(None, description="Previous convictions summary if present; otherwise null.")
    Tien_Su: Optional[str] = Field(None, description="Prior criminal history details if present; otherwise null.")
    Chi_Tiet_Nhan_Than: Optional[str] = Field(None, description="Details about relatives or next-of-kin if present; otherwise null.")
    Ngay_Tam_Giam: Optional[str] = Field(None, description="Date of temporary detention if present; otherwise null.")
    Trang_Thai_Co_Mat: Optional[str] = Field(None, description="Presence status at trial (Có mặt/Tạm vắng) if present; otherwise null.")


class PhanQuyetToaSoTham(BaseModel):
    Bi_Cao: str = Field(description="Defendant name the court decision applies to; must match a ThongTinBiCao entry.")
    Can_Cu_Dieu_Luat: List[CanCuDieuLuat] = Field(default_factory=list, description="List of legal citations used in the court's decision.")
    Pham_Toi: List[str] = Field(default_factory=list, description="List of offense names mentioned in the decision; may contain multiple distinct offense strings.")
    Phat_Tu: Optional[str] = Field(None, description="Imprisonment sentence text imposed by the court; keep concrete final total imprisonment length if present.")
    Phat_Tien: Optional[str] = Field(None, description="Monetary fine text imposed by the court; otherwise null.")
    An_Phi: Optional[str] = Field(None, description="Court fee decision text if present; otherwise null.")
    Hinh_Phat_Bo_Sung: Optional[str] = Field(None, description="Any supplementary punishments applied by the court; otherwise null.")
    Hinh_Phat_Bo_Sung_index: Optional[int] = Field(None, description="Index of the supplementary punishments in the Nhan_dinh_cua_toa_an section; If there is no supplementary punishments, return null.")
    Trach_Nhiem_Dan_Su: Optional[str] = Field(None, description="Civil liability decision text if present; otherwise null.")
    Tang_nang: Optional[str] = Field(None, description="Aggravating circumstances text if present; otherwise null.")
    Tang_nang_index: Optional[int] = Field(None, description="Index of the aggravating circumstances in the Nhan_dinh_cua_toa_an section; If there is no aggravating circumstances, return null.")
    Giam_nhe: Optional[str] = Field(None, description="Mitigating circumstances text if present; otherwise null.")
    Giam_nhe_index: Optional[int] = Field(None, description="Index of the mitigating circumstances in the Nhan_dinh_cua_toa_an section; If there is no mitigating circumstances, return null.")

class DeNghiVKS(BaseModel):
    Bi_Cao: str = Field(description="Defendant name the prosecutor's recommendation applies to; must match a ThongTinBiCao entry.")
    Pham_Toi: List[str] = Field(default_factory=list, description="List of offense names mentioned in the decision; may contain multiple distinct offense strings.")
    Phat_Tu: Optional[str] = Field(None, description="Imprisonment sentence text imposed by the court; keep concrete final total imprisonment length if present.")
    
class LLMExtractionOutput(BaseModel):
    Thong_Tin_Bi_Cao: List[ThongTinBiCao]
    De_Nghi_Cua_Vien_Kiem_Sat: List[DeNghiVKS]
    PHAN_QUYET_CUA_TOA_SO_THAM: List[PhanQuyetToaSoTham]
    Xu_Ly_Vat_Chung: Optional[str] = Field(None, description="How physical evidence was handled according to the case-level decision; one value per case, not per defendant.")


class VerdictOnlyOutput(BaseModel):
    PHAN_QUYET_CUA_TOA_SO_THAM: List[PhanQuyetToaSoTham]


def build_json_schema_prompt(schema: type[BaseModel]) -> str:
    """Render a prompt-friendly JSON skeleton that includes per-field instructions.

    This uses the project's RAG helper so output formatting matches other prompts.
    The returned value is a pretty JSON string ready to be inserted into system prompts.
    """
    skeleton = build_output_schema_instruction(schema)
    return json.dumps(skeleton, ensure_ascii=False, indent=2)
