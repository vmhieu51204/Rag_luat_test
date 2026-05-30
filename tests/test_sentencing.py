from __future__ import annotations

import unittest

from rag.core.sentencing import LIFE_IMPRISONMENT_MONTHS, extract_imprisonment_months


class SentencingTests(unittest.TestCase):
    def test_combined_life_sentence_inside_parentheses_overrides_new_sentence(self):
        sentence = (
            "12 năm tù (Hình phạt cho tội mới, sau đó tổng hợp với bản án tù chung thân trước đó "
            "theo Điều 56 BLHS, hình phạt chung là Tù chung thân)"
        )

        self.assertEqual(extract_imprisonment_months(sentence), LIFE_IMPRISONMENT_MONTHS)

    def test_combined_life_sentence_after_new_sentence_overrides_new_sentence(self):
        sentence = "09 năm tù, tổng hợp với án chung thân, hình phạt chung là tù chung thân"

        self.assertEqual(extract_imprisonment_months(sentence), LIFE_IMPRISONMENT_MONTHS)


if __name__ == "__main__":
    unittest.main()
