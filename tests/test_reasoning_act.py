from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag.core.law_retriever import LawClauseRetriever
from rag.generation.reasoning_act import (
    classify_supporting_article_by_facts,
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


class ReasoningActTests(unittest.TestCase):
    def test_exact_law_retrieval_records_found_and_missing(self):
        retriever = LawClauseRetriever("raw_law.json")
        articles = retrieve_law_articles(["201", "201-2", "51-1-s", "52-1-g", "47", "999999"], retriever)
        by_signature = {item.signature: item for item in articles}

        for signature in ["201", "201-2", "51-1-s", "52-1-g", "47"]:
            self.assertIn(signature, by_signature)
            self.assertTrue(by_signature[signature].found, signature)
            self.assertTrue(by_signature[signature].text)

        self.assertFalse(by_signature["999999"].found)
        self.assertEqual(by_signature["999999"].missing_reason, "dieu_not_found")

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


if __name__ == "__main__":
    unittest.main()
