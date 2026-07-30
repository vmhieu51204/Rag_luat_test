## Plan: Greedy Set Cover for Clause Matching

Implement a standalone CLI script at the repository root to run greedy set cover for each test case in synth/split against all train cases, using full clause signatures (dieu-khoan-diem), fixed top-k selection, and JSON output aligned with current evaluation conventions.

**Steps**
1. Reuse clause extraction semantics from [evaluate.py](evaluate.py#L56) so dict/list/scalar variants of Cac_Dieu_Quyet_Dinh are handled consistently and converted to signature sets.
2. Build dataset loaders for:
   1. train subsets from /home/hieujayce/Downloads/complete_repo/chunk/Chuong_XXII_chunked/train
   2. query universes from /home/hieujayce/Downloads/complete_repo/chunk/Chuong_XXII_chunked/synth/split  
   Each case uses Ma_Ban_An (fallback filename stem) and one deduplicated clause-signature set.
3. Implement greedy core per query case:
   1. Initialize uncovered = query clauses
   2. At each iteration choose the train case with maximum gain (newly covered clauses)
   3. Deterministic tie-breaker: higher gain, then smaller train subset size, then lexicographic case id
   4. Continue until fixed top-k picks are made
4. Record per-iteration trace for each test case:
   1. selected train case id
   2. newly covered clauses
   3. cumulative covered count and coverage ratio
   4. remaining uncovered clauses
5. Build final JSON report (config + summary + per_test) following existing style from [evaluate.py](evaluate.py#L310) and [output/eval_results_notebook.json](output/eval_results_notebook.json).
6. Add CLI options:
   1. --train_dir (default: train folder above)
   2. --test_dir (default: synth/split folder above)
   3. --top_k (default 5)
   4. --output (default /home/hieujayce/Downloads/complete_repo/output/greedy_set_cover_results.json)
7. Validate determinism and correctness by running twice and confirming identical selections/coverage traces for sampled test cases.

**Relevant files**
- New root script to create: greedy_set_cover.py
- Reuse extraction behavior from [evaluate.py](evaluate.py#L56)
- Reuse loading/result-shape conventions from [evaluate.py](evaluate.py#L77) and [evaluate.py](evaluate.py#L310)
- Align summary style with [output/eval_results_notebook.json](output/eval_results_notebook.json)

**Verification**
1. Run: python /home/hieujayce/Downloads/complete_repo/greedy_set_cover.py --top_k 5
2. Validate JSON: python -m json.tool /home/hieujayce/Downloads/complete_repo/output/greedy_set_cover_results.json
3. Spot-check one case: confirm cumulative coverage is non-decreasing and each iteration gain equals set intersection delta.
4. Re-run and verify deterministic output equality for selected train case order and gain values.

**Decisions Captured**
- Clause granularity: full signatures (dieu-khoan-diem)
- Selection rule: fixed top-k train cases
- Output: JSON with per-test selected train cases and coverage stats
- Out of scope: embedding/text similarity retrieval, weighted set cover, reranking by narrative text

If this plan looks right, approve and I’ll hand it off for implementation.