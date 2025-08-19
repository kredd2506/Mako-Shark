# orchestrator/router.py
import re
from core.llm import chat

ROUTER_SYS = """
You route user intents into one of: K8S, MONITOR, WEB.
- K8S: kubectl-like queries (list/describe/permissions/ns)
- MONITOR: GPU/DCGM/Prometheus stats or Kubernetes concepts table or doc KB search
- WEB: external web research/questions "how to" that require sources
Respond ONLY as JSON with fields: {"route":"K8S|MONITOR|WEB","reason":"..."}
"""

HARD_HINTS = {
    'pods': 'K8S', 'deploy': 'K8S', 'service ': 'K8S', 'ingress': 'K8S', 'replicaset': 'K8S', 'statefulset': 'K8S',
    'gpu': 'MONITOR', 'promql': 'MONITOR', 'prometheus': 'MONITOR', 'dcgm': 'MONITOR', 'utilization': 'MONITOR',
    'docs': 'MONITOR', 'concepts': 'MONITOR', 'table': 'MONITOR',
    'search': 'WEB', 'how to': 'WEB', 'tutorial': 'WEB', 'kubernetes.io': 'WEB',
    'prometheus': 'MONITOR', 'promql': 'MONITOR', 'prom_': 'MONITOR', 'prom ': 'MONITOR',
}

def route(user_text: str) -> str:
    # quick heuristic first
    low = user_text.lower()
    for k, v in HARD_HINTS.items():
        if k in low:
            return v
    # ask LLM if ambiguous
    msg = [
        {"role": "system", "content": ROUTER_SYS},
        {"role": "user", "content": user_text}
    ]
    out = chat(msg, temperature=0)
    import json
    try:
        r = json.loads(out)
        return r.get("route", "K8S")
    except Exception:
        return "K8S"
