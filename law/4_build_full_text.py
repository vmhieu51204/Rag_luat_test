import json

class Stage4_FullTextBuilder:
    """
    Builds two new fields on every Điều, Khoản and Điểm node:

    • full_text  (on Điều and Khoản only)
        Own text  +  the prefixed full_text/text of every direct child,
        recursively.  Represents the complete subtree rooted at this node.

    • text  (overwritten on Khoản and Điểm)
        The node's original text  PREPENDED with the text of every ancestor
        up to and including its parent Điều.
        Rule: a Điểm gets  "Điều text  →  Khoản text  →  Điểm own text".
              a Khoản gets "Điều text  →  Khoản own text".
        Sibling branches are NOT included.
    """

    # ------------------------------------------------------------------ #
    #  Bottom-up pass: build full_text on Điều / Khoản                    #
    # ------------------------------------------------------------------ #

    def _full_text_khoan(self, khoan):
        """full_text of a Khoản = its own text + each Điểm prefixed."""
        parts = [khoan["text"]]
        for diem in khoan.get("Điểm", []):
            parts.append(f"{diem['index']}) {diem['text']}")
        khoan["full_text"] = "\n".join(parts)
        return khoan["full_text"]

    def _full_text_dieu(self, dieu):
        """full_text of a Điều = its own text + each Khoản (full) prefixed."""
        parts = [dieu["text"]]
        for khoan in dieu.get("Khoản", []):
            khoan_full = self._full_text_khoan(khoan)
            parts.append(f"{khoan['index']}. {khoan_full}")
        dieu["full_text"] = "\n".join(parts)
        return dieu["full_text"]

    # ------------------------------------------------------------------ #
    #  Top-down pass: rewrite text on Khoản / Điểm with ancestor context  #
    # ------------------------------------------------------------------ #

    def _set_inherited_text(self, dieu):
        """
        Overwrite the `text` field of each Khoản and Điểm so that it
        contains the text of all ancestors down to (and including) itself.
        """
        dieu_text = dieu["text"]

        for khoan in dieu.get("Khoản", []):
            own_khoan = khoan["text"]                       # save original
            # Khoản.text = Điều text + Khoản own text
            khoan["text"] = "\n".join([dieu_text, f"{khoan['index']}. {own_khoan}"])

            for diem in khoan.get("Điểm", []):
                own_diem = diem["text"]                     # save original
                # Điểm.text = Điều text + Khoản own text + Điểm own text
                diem["text"] = "\n".join([
                    dieu_text,
                    f"{khoan['index']}. {own_khoan}",
                    f"{diem['index']}) {own_diem}",
                ])

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def process(self, document_data):
        for chuong in document_data:
            for dieu in chuong.get("Điều", []):
                # 1. Build full_text bottom-up (uses original text values)
                self._full_text_dieu(dieu)
                # 2. Rewrite text top-down with ancestor context
                self._set_inherited_text(dieu)
        return document_data


if __name__ == "__main__":
    input_file  = r"law/stage3_final.json"
    output_file = r"law/stage4_full_text.json"

    try:
        print("Stage 4: Building full_text fields...")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        builder = Stage4_FullTextBuilder()
        result = builder.process(data)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        print(f"✅ Success! Saved enriched JSON to {output_file}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")