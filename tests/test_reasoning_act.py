from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag.core.law_retriever import LawClauseRetriever
from rag.generation.reasoning_act import (
    classify_supporting_article_by_facts,
    compact_law_signatures,
    retrieve_sentencing_calibration_cases,
    retrieve_law_articles,
    retrieve_similar_cases,
)


class FakeRuntime:
    def query_train(self, *, query_text, top_k, exclude_doc_id=None, include=None):
        return {
            "metadatas": [[
                {"doc_id": "case-a"},
                {"doc_id": "case-a"},
                {"doc_id": "case-b"},
                {"doc_id": "case-c"},
            ]],
            "distances": [[0.10, 0.20, 0.05, 0.30]],
        }


class FakeCalibrationRuntime:
    def query_train(self, *, query_text, top_k, exclude_doc_id=None, include=None):
        if "thành khẩn" in query_text.lower():
            return {
                "metadatas": [[
                    {
                        "doc_id": "case-a",
                        "field": "PHAN_QUYET_CUA_TOA_SO_THAM.Giam_nhe",
                        "record_index": 0,
                        "defendant_name": "A",
                    },
                    {
                        "doc_id": "case-b",
                        "field": "PHAN_QUYET_CUA_TOA_SO_THAM.Giam_nhe",
                        "record_index": 0,
                        "defendant_name": "B",
                    },
                ]],
                "distances": [[0.10, 0.20]],
            }
        return {
            "metadatas": [[
                {
                    "doc_id": "case-c",
                    "field": "PHAN_QUYET_CUA_TOA_SO_THAM.Tang_nang",
                    "record_index": 0,
                    "defendant_name": "C",
                }
            ]],
            "distances": [[0.05]],
        }


class ManyCalibrationRuntime:
    def query_train(self, *, query_text, top_k, exclude_doc_id=None, include=None):
        is_mitigation = "mitigation" in query_text
        prefix = "mit" if is_mitigation else "agg"
        field = (
            "PHAN_QUYET_CUA_TOA_SO_THAM.Giam_nhe"
            if is_mitigation
            else "PHAN_QUYET_CUA_TOA_SO_THAM.Tang_nang"
        )
        metadatas = [
            {
                "doc_id": f"{prefix}-{idx}",
                "field": field,
                "record_index": 0,
                "defendant_name": f"{prefix.upper()}{idx}",
            }
            for idx in range(8)
        ]
        return {
            "metadatas": [metadatas],
            "distances": [[idx / 100 for idx in range(len(metadatas))]],
        }


class ReasoningActTests(unittest.TestCase):
    def test_exact_law_retrieval_records_found_and_missing(self):
        retriever = LawClauseRetriever("raw_law.json")
        articles = retrieve_law_articles(["201-2", "51-1-s", "52-1-g", "47", "999999"], retriever)
        by_signature = {item.signature: item for item in articles}

        for signature in ["201-2", "51-1-s", "52-1-g", "47"]:
            self.assertIn(signature, by_signature)
            self.assertTrue(by_signature[signature].found, signature)
            self.assertTrue(by_signature[signature].text)

        self.assertFalse(by_signature["999999"].found)
        self.assertEqual(by_signature["999999"].missing_reason, "dieu_not_found")

    def test_law_retrieval_compacts_subclauses_when_full_article_requested(self):
        self.assertEqual(compact_law_signatures(["201-2", "201", "201-1-a", "51-1-s"]), ["201", "51-1-s"])

        articles = retrieve_law_articles(["201-2", "201", "201-1-a"], LawClauseRetriever("raw_law.json"))

        self.assertEqual([item.signature for item in articles], ["201"])
        self.assertEqual(articles[0].level, "dieu")
        self.assertTrue(articles[0].found)

    def test_supporting_article_default_classification(self):
        found = retrieve_law_articles(["47", "53", "58"], LawClauseRetriever("raw_law.json"))
        by_signature = {item.signature: item for item in found}

        article_53 = classify_supporting_article_by_facts(
            article="53",
            retrieved=by_signature["53"],
            case_text='{"Tien_An": "đã bị kết án, chưa xóa án tích"}',
        )
        article_58 = classify_supporting_article_by_facts(
            article="58",
            retrieved=by_signature["58"],
            case_text="Có đồng phạm giúp sức trong quá trình thực hiện hành vi.",
        )
        article_47 = classify_supporting_article_by_facts(
            article="47",
            retrieved=by_signature["47"],
            case_text="Bị cáo thành khẩn khai báo, ăn năn hối cải.",
        )

        self.assertEqual(article_53.status, "fact_dependent")
        self.assertEqual(article_58.status, "fact_dependent")
        self.assertEqual(article_47.status, "not_applicable")

    def test_similar_case_filtering_deduplicates_and_filters_by_selected_dieu(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = Path(tmp)
            docs = {
                "case-a": {
                    "Ma_Ban_An": "case-a",
                    "Summary": "Lừa đảo chiếm đoạt tài sản bằng thông tin gian dối.",
                    "PHAN_QUYET_CUA_TOA_SO_THAM": [
                        {
                            "Bi_Cao": "A",
                            "Can_Cu_Dieu_Luat": [{"Dieu": "174", "Bo_Luat_Va_Van_Ban_Khac": "BLHS"}],
                            "Pham_Toi": ["Lừa đảo chiếm đoạt tài sản"],
                            "Phat_Tu": "02 năm tù",
                            "Giam_nhe": "Thành khẩn khai báo",
                        }
                    ],
                },
                "case-b": {
                    "Ma_Ban_An": "case-b",
                    "Summary": "Trộm cắp tài sản.",
                    "PHAN_QUYET_CUA_TOA_SO_THAM": [
                        {
                            "Bi_Cao": "B",
                            "Can_Cu_Dieu_Luat": [{"Dieu": "173", "Bo_Luat_Va_Van_Ban_Khac": "BLHS"}],
                            "Pham_Toi": ["Trộm cắp tài sản"],
                            "Phat_Tu": "01 năm tù",
                        }
                    ],
                },
                "case-c": {
                    "Ma_Ban_An": "case-c",
                    "Summary": "Lừa đảo chiếm đoạt tài sản, đã bồi thường.",
                    "PHAN_QUYET_CUA_TOA_SO_THAM": [
                        {
                            "Bi_Cao": "C",
                            "Can_Cu_Dieu_Luat": [{"Dieu": "174", "Bo_Luat_Va_Van_Ban_Khac": "BLHS"}],
                            "Pham_Toi": ["Lừa đảo chiếm đoạt tài sản"],
                            "Phat_Tu": "03 năm tù",
                            "Tang_nang": "Phạm tội nhiều lần",
                        }
                    ],
                },
            }
            for doc_id, data in docs.items():
                (train_dir / f"{doc_id}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            index = {
                "case-a": {"dieu_only": {"174"}, "full_signature": {"174"}},
                "case-b": {"dieu_only": {"173"}, "full_signature": {"173"}},
                "case-c": {"dieu_only": {"174"}, "full_signature": {"174"}},
            }
            similar = retrieve_similar_cases(
                runtime=FakeRuntime(),
                train_dir=train_dir,
                train_articles_index=index,
                query_text="lừa đảo chiếm đoạt tài sản thành khẩn",
                selected_dieu="174",
                exclude_doc_id="test",
                broad_top_k=64,
                top_k=5,
            )

        self.assertEqual(len(similar), 2)
        self.assertEqual({item.doc_id for item in similar}, {"case-a", "case-c"})
        self.assertTrue(all(item.sentence for item in similar))

    def test_sentencing_calibration_retrieval_uses_verdict_factor_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = Path(tmp)
            docs = {
                "case-a": {
                    "Ma_Ban_An": "case-a",
                    "De_Nghi_Cua_Vien_Kiem_Sat": [{"Bi_Cao": "A", "Phat_Tu": "từ 02 năm đến 03 năm tù"}],
                    "PHAN_QUYET_CUA_TOA_SO_THAM": [
                        {
                            "Bi_Cao": "A",
                            "Phat_Tu": "02 năm tù",
                            "Giam_nhe": "Thành khẩn khai báo",
                            "Tang_nang": None,
                        }
                    ],
                },
                "case-b": {
                    "Ma_Ban_An": "case-b",
                    "De_Nghi_Cua_Vien_Kiem_Sat": [{"Bi_Cao": "B", "Phat_Tu": "từ 01 năm đến 02 năm tù"}],
                    "PHAN_QUYET_CUA_TOA_SO_THAM": [
                        {
                            "Bi_Cao": "B",
                            "Phat_Tu": "01 năm tù",
                            "Giam_nhe": "Thành khẩn khai báo, ăn năn hối cải",
                        }
                    ],
                },
                "case-c": {
                    "Ma_Ban_An": "case-c",
                    "De_Nghi_Cua_Vien_Kiem_Sat": [{"Bi_Cao": "C", "Phat_Tu": "từ 03 năm đến 04 năm tù"}],
                    "PHAN_QUYET_CUA_TOA_SO_THAM": [
                        {
                            "Bi_Cao": "C",
                            "Phat_Tu": "03 năm tù",
                            "Tang_nang": "Phạm tội nhiều lần",
                        }
                    ],
                },
            }
            for doc_id, data in docs.items():
                (train_dir / f"{doc_id}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            cases = retrieve_sentencing_calibration_cases(
                runtime=FakeCalibrationRuntime(),
                train_dir=train_dir,
                train_articles_index={
                    "case-a": {"dieu_only": {"174"}, "full_signature": {"174"}},
                    "case-b": {"dieu_only": {"173"}, "full_signature": {"173"}},
                    "case-c": {"dieu_only": {"174"}, "full_signature": {"174"}},
                },
                mitigation_factors=["Bị cáo thành khẩn khai báo"],
                aggravation_factors=["Bị cáo phạm tội nhiều lần"],
                selected_dieu="174",
                exclude_doc_id="test",
                top_k_per_factor=1,
                broad_top_k=4,
            )

        self.assertEqual(len(cases), 2)
        mitigation = next(item for item in cases if item.factor_type == "mitigation")
        aggravation = next(item for item in cases if item.factor_type == "aggravation")
        self.assertEqual(mitigation.doc_id, "case-a")
        self.assertEqual(mitigation.prosecution_proposal["Phat_Tu"], "từ 02 năm đến 03 năm tù")
        self.assertEqual(mitigation.prosecution_proposal["Phat_Tu_month_range"], [24, 36])
        self.assertEqual(mitigation.court_sentence, "02 năm tù")
        self.assertEqual(aggravation.doc_id, "case-c")
        self.assertEqual(aggravation.prosecution_proposal["Phat_Tu_month_range"], [36, 48])
        self.assertEqual(aggravation.court_aggravation, "Phạm tội nhiều lần")

    def test_sentencing_calibration_caps_final_cases_per_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = Path(tmp)
            train_articles_index = {}
            for prefix, field_name in (("mit", "Giam_nhe"), ("agg", "Tang_nang")):
                for idx in range(8):
                    doc_id = f"{prefix}-{idx}"
                    train_articles_index[doc_id] = {"dieu_only": {"174"}, "full_signature": {"174"}}
                    (train_dir / f"{doc_id}.json").write_text(
                        json.dumps(
                            {
                                "Ma_Ban_An": doc_id,
                                "PHAN_QUYET_CUA_TOA_SO_THAM": [
                                    {
                                        "Bi_Cao": f"{prefix.upper()}{idx}",
                                        "Phat_Tu": "02 năm tù",
                                        field_name: f"{prefix} factor {idx}",
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )

            cases = retrieve_sentencing_calibration_cases(
                runtime=ManyCalibrationRuntime(),
                train_dir=train_dir,
                train_articles_index=train_articles_index,
                mitigation_factors=["mitigation factor 1", "mitigation factor 2"],
                aggravation_factors=["aggravation factor 1", "aggravation factor 2"],
                selected_dieu="174",
                exclude_doc_id="test",
                top_k_per_factor=3,
                broad_top_k=8,
            )

        mitigation_cases = [item for item in cases if item.factor_type == "mitigation"]
        aggravation_cases = [item for item in cases if item.factor_type == "aggravation"]
        self.assertLessEqual(len(mitigation_cases), 5)
        self.assertLessEqual(len(aggravation_cases), 5)


if __name__ == "__main__":
    unittest.main()
