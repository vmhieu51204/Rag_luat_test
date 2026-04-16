import json
from pathlib import Path

TRAIN_DIR       = "./chunk/Chuong_XXII_chunked/train"
TEST_DIR        = "./chunk/Chuong_XXII_chunked/synth/split"

ID_FIELD       = "Ma_Ban_An"
ARTICLES_FIELD = "Cac_Dieu_Quyet_Dinh"

def extract_article_signatures(articles) -> set:
    sigs = set()
    if isinstance(articles, dict):
        for val_list in articles.values():
            if isinstance(val_list, list):
                for item in val_list:
                    if isinstance(item, list):
                        sigs.add("-".join(str(i) for i in item))
                    else:
                        sigs.add(str(item))
            else:
                sigs.add(str(val_list))
    elif isinstance(articles, list):
        for a in articles:
            if isinstance(a, list):
                sigs.add("-".join(str(i) for i in a))
            else:
                sigs.add(str(a))
    elif articles:
        sigs.add(str(articles))
    return sigs

def load_articles_index(raw_dir: Path) -> dict:
    index = {}
    for f in raw_dir.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        doc_id = data.get(ID_FIELD, f.stem)
        articles = data.get(ARTICLES_FIELD)
        if articles:
            index[doc_id] = extract_article_signatures(articles)
    return index

train_path = Path(TRAIN_DIR)
test_path  = Path(TEST_DIR)

train_articles_index = load_articles_index(train_path)
test_articles_index = load_articles_index(test_path)

all_train_clauses = set()
for clauses in train_articles_index.values():
    all_train_clauses.update(clauses)

all_test_clauses = set()
for clauses in test_articles_index.values():
    all_test_clauses.update(clauses)

unique_to_test = all_test_clauses - all_train_clauses

print(f"Total unique clauses in TRAIN_DIR: {len(all_train_clauses)}")
print(f"Total unique clauses in TEST_DIR: {len(all_test_clauses)}")
print(f"Number of clauses in TEST_DIR but not in TRAIN_DIR: {len(unique_to_test)}")
if len(unique_to_test) > 0:
    print(f"Clauses in TEST but not in TRAIN: {sorted(list(unique_to_test))}")
