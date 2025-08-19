# experiments/datasets.py
import os, json
from typing import List, Dict, Any

# Expected formats (minimal):
# FanOutQA/FRAMES jsonl: {"id": "...", "question": "...", "answer": "..."}
# AssistantBench json: {"tasks": [{"id": "...", "confirmed_task": "...", "answer": "..."}]}


def _fallback_samples() -> List[Dict[str, Any]]:
    # Tiny built‑in samples so the harness runs even without files
    return [
        {"id":"ab_05", "question":"Return the Prometheus HTTP endpoint used for an instant query and the required parameter name as {endpoint, param}.",
         "answer":"{endpoint: /api/v1/query, param: query}"},
        {"id":"mako_tc_0001", "question":"Which Kubernetes resource ensures stable network identity and persistent storage for each replica, and what is its default update strategy?",
         "answer":"statefulset rollingupdate"},
    ]


def load_fanoutqa(path: str = None, limit: int = 50):
    items = []
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit: break
                obj = json.loads(line)
                items.append({"id": obj.get("id", f"ex_{i}"), "question": obj["question"], "answer": obj.get("answer", "")})
    else:
        items = _fallback_samples()[:limit]
    return items


def load_frames(path: str = None, limit: int = 50):
    return load_fanoutqa(path, limit)


def load_assistantbench(path: str = None, limit: int = 50):
    tasks = []
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for t in data.get("tasks", [])[:limit]:
                tasks.append({"id": t.get("task_id", t.get("id")), "confirmed_task": t.get("confirmed_task"), "answer": t.get("answer", "")})
    else:
        # Convert fallbacks into AB‑style
        for s in _fallback_samples()[:limit]:
            tasks.append({"id": s["id"], "confirmed_task": s["question"], "answer": s["answer"]})
    return tasks