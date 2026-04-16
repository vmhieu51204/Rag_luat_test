import json

notebook_path = "evaluate_notebook.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the index of the Summary cell to insert before it
# or just append to the end.
# Looking at the previous tail, let's find the cell with " ## 18. Summary"
summary_index = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and any(" ## 18. Summary" in line for line in cell["source"]):
        summary_index = i
        break

new_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        " ## 17. Law Clauses Analysis (TEST vs TRAIN)\n",
        "\n",
        " Identification of clauses present in the test set but entirely missing from the training set index."
    ]
}

new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "train_path = Path(TRAIN_DIR)\n",
        "test_path  = Path(TEST_DIR)\n",
        "\n",
        "train_articles_index = load_articles_index(train_path)\n",
        "test_articles_index = load_articles_index(test_path)\n",
        "\n",
        "all_train_clauses = set()\n",
        "for clauses in train_articles_index.values():\n",
        "    all_train_clauses.update(clauses)\n",
        "\n",
        "all_test_clauses = set()\n",
        "for clauses in test_articles_index.values():\n",
        "    all_test_clauses.update(clauses)\n",
        "\n",
        "unique_to_test = all_test_clauses - all_train_clauses\n",
        "\n",
        "print(f\"Total unique clauses in TRAIN_DIR: {len(all_train_clauses)}\")\n",
        "print(f\"Total unique clauses in TEST_DIR: {len(all_test_clauses)}\")\n",
        "print(f\"Number of law clauses in TEST_DIR but not in TRAIN_DIR: {len(unique_to_test)}\")\n",
        "if len(unique_to_test) > 0:\n",
        "    print(f\"\\nClauses present in TEST but missing from TRAIN index:\\n{sorted(list(unique_to_test))}\")"
    ]
}

if summary_index != -1:
    nb["cells"].insert(summary_index, new_markdown_cell)
    nb["cells"].insert(summary_index + 1, new_code_cell)
else:
    nb["cells"].append(new_markdown_cell)
    nb["cells"].append(new_code_cell)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Successfully added the analysis cells to evaluate_notebook.ipynb")
