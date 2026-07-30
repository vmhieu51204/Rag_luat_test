uv run python -m rag.evaluation.reasoning_one_call_eval \
  --test-dir chunk/test/2024 \
  --results-out output/reasoning_one_call_eval/results_2024.json

uv run python -m rag.evaluation.reasoning_act_eval \
  --test-dir chunk/test/2024 \
  --train-dir chunk/train \
  --law-json raw_law.json \
  --case-db-dir output/reasoning_act_eval/case_db \
  --results-out output/reasoning_act_eval/results_2024.json

uv run python -m rag.evaluation.reasoning_past_eval \
  --test-dir chunk/test/2024 \
  --train-dir chunk/train \
  --law-json raw_law.json \
  --case-db-dir output/generation_eval/case_db \
  --law-db-dir output/generation_eval/law_db \
  --results-out output/generation_eval/verdict_generation_system_eval_2024.json
