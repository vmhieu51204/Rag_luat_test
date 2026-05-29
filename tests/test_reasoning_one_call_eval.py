from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from rag.evaluation.reasoning_one_call_eval import GenerationOutputNoAnalysis, _evaluate_single_doc
from rag.generation.schemas import GenerationOutput, PredictedDefendant, PredictedLawClause
from rag.llm.providers import LLMProvider


class ReasoningOneCallEvalTests(unittest.TestCase):
    def test_one_call_eval_uses_single_generic_llm_call_and_scores_output(self):
        data = {
            "Ma_Ban_An": "case-1",
            "THONG_TIN_CHUNG": {"Thong_Tin_Bi_Cao": [{"Ho_Ten": "Nguyen Van A"}]},
            "Synthetic_summary": "Nguyen Van A trộm cắp tài sản trị giá 5.000.000 đồng.",
            "PHAN_QUYET_CUA_TOA_SO_THAM": [
                {
                    "Bi_Cao": "Nguyen Van A",
                    "Pham_Toi": "Trộm cắp tài sản",
                    "Can_Cu_Dieu_Luat": [
                        {
                            "Dieu": "173",
                            "Khoan": "1",
                            "Bo_Luat_Va_Van_Ban_Khac": "BLHS",
                        }
                    ],
                    "Phat_Tu": "06 tháng tù",
                }
            ],
        }
        predicted = GenerationOutput(
            defendants=[
                PredictedDefendant(
                    Bi_Cao="Nguyen Van A",
                    Toi_Danh="Trộm cắp tài sản",
                    Applied_Law_Clauses=[
                        PredictedLawClause(
                            Dieu="173",
                            Khoan="1",
                            Bo_Luat_Va_Van_Ban_Khac="BLHS",
                            Tinh_tiet_ap_dung="Trộm cắp tài sản trị giá 5.000.000 đồng.",
                        )
                    ],
                    Phat_Tu="06 tháng tù",
                )
            ]
        )
        call_count = 0

        def fake_generate_structured_output(**kwargs):
            nonlocal call_count
            call_count += 1
            self.assertEqual(kwargs["output_model"], GenerationOutput)
            self.assertIn("case_fields", kwargs["user_prompt"])
            return predicted, {"provider": "openrouter", "model": "test-model"}

        with patch(
            "rag.evaluation.reasoning_one_call_eval.generate_structured_output",
            side_effect=fake_generate_structured_output,
        ):
            result = _evaluate_single_doc(
                path=Path("case-1.json"),
                data=data,
                input_fields=["THONG_TIN_CHUNG.Thong_Tin_Bi_Cao", "Synthetic_summary"],
                provider=LLMProvider.OPENROUTER,
                model_name="test-model",
                only_blhs=True,
                use_provider_fallback=False,
                omit_phan_tich_phap_ly=False,
            )

        self.assertEqual(call_count, 1)
        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["trace"]["mode"], "one_call_generic")
        self.assertFalse(result["trace"]["agentic_reasoning"])
        self.assertFalse(result["trace"]["law_retrieval"])
        self.assertEqual(result["doc_metrics"]["law_clause_f1_macro"], 1.0)
        self.assertEqual(result["doc_metrics"]["phat_tu_rmse_months"], 0.0)

    def test_omit_phan_tich_phap_ly_uses_no_analysis_schema(self):
        data = {
            "Ma_Ban_An": "case-1",
            "THONG_TIN_CHUNG": {"Thong_Tin_Bi_Cao": [{"Ho_Ten": "Nguyen Van A"}]},
            "Synthetic_summary": "Nguyen Van A trộm cắp tài sản trị giá 5.000.000 đồng.",
            "PHAN_QUYET_CUA_TOA_SO_THAM": [
                {
                    "Bi_Cao": "Nguyen Van A",
                    "Pham_Toi": "Trộm cắp tài sản",
                    "Can_Cu_Dieu_Luat": [
                        {
                            "Dieu": "173",
                            "Khoan": "1",
                            "Bo_Luat_Va_Van_Ban_Khac": "BLHS",
                        }
                    ],
                    "Phat_Tu": "06 tháng tù",
                }
            ],
        }
        predicted = GenerationOutputNoAnalysis.model_validate(
            {
                "defendants": [
                    {
                        "Bi_Cao": "Nguyen Van A",
                        "Toi_Danh": "Trộm cắp tài sản",
                        "Applied_Law_Clauses": [
                            {
                                "Dieu": "173",
                                "Khoan": "1",
                                "Bo_Luat_Va_Van_Ban_Khac": "BLHS",
                                "Tinh_tiet_ap_dung": "Trộm cắp tài sản trị giá 5.000.000 đồng.",
                            }
                        ],
                        "Phat_Tu": "06 tháng tù",
                    }
                ]
            }
        )

        def fake_generate_structured_output(**kwargs):
            self.assertEqual(kwargs["output_model"], GenerationOutputNoAnalysis)
            self.assertNotIn("Phan_Tich_Phap_Ly", kwargs["user_prompt"])
            return predicted, {"provider": "openrouter", "model": "test-model"}

        with patch(
            "rag.evaluation.reasoning_one_call_eval.generate_structured_output",
            side_effect=fake_generate_structured_output,
        ):
            result = _evaluate_single_doc(
                path=Path("case-1.json"),
                data=data,
                input_fields=["THONG_TIN_CHUNG.Thong_Tin_Bi_Cao", "Synthetic_summary"],
                provider=LLMProvider.OPENROUTER,
                model_name="test-model",
                only_blhs=True,
                use_provider_fallback=False,
                omit_phan_tich_phap_ly=True,
            )

        self.assertTrue(result["trace"]["omit_phan_tich_phap_ly"])
        self.assertIsNone(result["prediction_raw"]["defendants"][0]["Phan_Tich_Phap_Ly"])
        self.assertEqual(result["doc_metrics"]["law_clause_f1_macro"], 1.0)


if __name__ == "__main__":
    unittest.main()
