# Experiments Harness (Demo Mode)

This folder lets you reproduce INFOGENT‑style evaluations *without modifying the main app*.

## Modes
- `--mode direct` → Wikipedia‑only search, aggregate snippets, ask LLM for final answer.
- `--mode interactive` → Simulated multi‑step navigation with action logs.

## Datasets
- **FanOutQA / FRAMES**: provide a JSONL file with `{id, question, answer}` per line.
- **AssistantBench**: provide a JSON file: `{ "tasks": [{ "task_id", "confirmed_task", "answer" }] }`.
- Sample stubs in `sample_datasets/` or the harness will fall back to tiny built‑ins.

## Env vars
- `NRP_API_KEY` (required): your NRP key for `core.llm.chat`.
- `SERPER_API_KEY` (optional): if set, `restricted_search` uses Serper; otherwise it falls back to Wikipedia API.
- `WIKI_ONLY=1` (recommended): force allowlist to `wikipedia.org` for apples‑to‑apples comparisons.

## Examples
```bash
python experiments/runner.py --mode direct --suite fanoutqa --data experiments/sample_datasets/fanoutqa_dev_sample.jsonl --limit 50
python experiments/runner.py --mode direct --suite frames --data experiments/sample_datasets/frames_dev_sample.jsonl --limit 50
python experiments/runner.py --mode interactive --suite assistantbench --data experiments/sample_datasets/assistantbench_dev_sample.json --limit 20