# experiments/runner.py
import os, json, argparse, time
from typing import List, Dict, Any
from core.llm import chat
from experiments.policies import wiki_search, wiki_fetch, restricted_search
from experiments.eval import rouge_scores, normalize_text, string_accuracy
from experiments.datasets import load_fanoutqa, load_frames, load_assistantbench

SYS_DIRECT = (
    "You are an information aggregator. You will be given snippets extracted from multiple web pages. "
    "Use ONLY those snippets to answer the user question. Quote numbers carefully. If unknown, say 'unknown'."
)

SYS_INTERACTIVE = (
    "You are a navigator-extractor. Given a user task and previous notes, return one of: "
    "SEARCH:<query> | CLICK:<index> | AGGREGATE | ANSWER:<final>. Keep steps short."
)

# ——— Direct API‑driven: single shot aggregate then answer ———

def run_direct(items: List[Dict[str, Any]], k: int = 5, sleep_s: float = 0.2):
    outputs = []
    for i, ex in enumerate(items, 1):
        q = ex["question"]
        # Wikipedia‑only search (or restricted via Serper if SERPER_API_KEY present)
        hits = wiki_search(q, k=k)
        snippets = []
        for h in hits:
            txt = wiki_fetch(h["title"]) or h.get("snippet", "")
            if txt:
                snippets.append(f"[PAGE: {h['title']}]\n{txt}")
        context = "\n\n".join(snippets[:k]) or "(no context)"
        msgs = [
            {"role": "system", "content": SYS_DIRECT},
            {"role": "user", "content": f"Question: {q}\n\nContext:\n{context}"}
        ]
        ans = chat(msgs, temperature=0.0)
        outputs.append({"id": ex.get("id", f"ex_{i}"), "question": q, "answer": ans, "gold": ex.get("answer")})
        time.sleep(sleep_s)
    return outputs

# ——— Simulated Interactive visual: short multi‑step loop ———

def run_interactive(items: List[Dict[str, Any]], k: int = 5, max_steps: int = 8, sleep_s: float = 0.2):
    traces = []
    for i, ex in enumerate(items, 1):
        q = ex["question"] if "question" in ex else ex.get("confirmed_task", "")
        memory: List[str] = []
        actions = {"SEARCH":0, "CLICK":0, "AGGREGATE":0, "GO BACK":0, "TYPE":0, "ENTER":0}
        search_results = restricted_search(q, k=k)  # follows WIKI_ONLY/DOMAIN allowlist
        actions["SEARCH"] += 1
        clicked_idx = 0
        final_answer = None
        for step in range(max_steps):
            # Aggregate current page text
            if 0 <= clicked_idx < len(search_results):
                url = search_results[clicked_idx].get("link")
                title = search_results[clicked_idx].get("title")
                txt = wiki_fetch(title) if os.getenv("WIKI_ONLY") else wiki_fetch(title) or ""
                if not txt:
                    # fall back to simple fetch if not wiki
                    from tools.web_tools import fetch_text
                    txt = fetch_text(url) or ""
                snippet = txt[:1500]
                memory.append(f"[PAGE: {title}] {snippet}")
                actions["AGGREGATE"] += 1

            # Ask the navigator what to do next given memory
            nav_msgs = [
                {"role": "system", "content": SYS_INTERACTIVE},
                {"role": "user", "content": f"Task: {q}\nNotes so far:\n" + "\n".join(memory[-3:])}
            ]
            decision = chat(nav_msgs, temperature=0.2).strip()
            if decision.startswith("ANSWER:"):
                final_answer = decision.split(":",1)[1].strip()
                break
            elif decision.startswith("SEARCH:"):
                query = decision.split(":",1)[1].strip()
                search_results = restricted_search(query, k=k)
                actions["SEARCH"] += 1
                clicked_idx = 0
            elif decision.startswith("CLICK:"):
                try:
                    idx = int(decision.split(":",1)[1].strip())
                except Exception:
                    idx = 0
                clicked_idx = max(0, min(len(search_results)-1, idx))
                actions["CLICK"] += 1
            else:
                # default: try next result
                clicked_idx = (clicked_idx + 1) % max(1, len(search_results))
                actions["CLICK"] += 1
            time.sleep(sleep_s)

        if final_answer is None:
            # Fall back: ask for final answer from memory
            final_answer = chat([
                {"role":"system","content": SYS_DIRECT},
                {"role":"user","content": f"Question: {q}\nContext:\n" + "\n\n".join(memory)}
            ], temperature=0.0)
        traces.append({
            "id": ex.get("id", f"ex_{i}"),
            "question": q,
            "answer": final_answer,
            "gold": ex.get("answer"),
            "actions": actions
        })
    return traces

# ——— Reporting ———

def report(results: List[Dict[str, Any]]):
    n = len(results)
    acc_cnt = 0
    r1 = r2 = rl = 0.0
    for r in results:
        pred = normalize_text(r.get("answer", ""))
        gold = normalize_text(r.get("gold", ""))
        if gold:
            acc_cnt += string_accuracy(pred, gold)
            R1, R2, RL = rouge_scores(pred, gold)
            r1 += R1; r2 += R2; rl += RL
    acc = (acc_cnt / n * 100.0) if n else 0.0
    avg_r1 = (r1 / n * 100.0) if n else 0.0
    avg_r2 = (r2 / n * 100.0) if n else 0.0
    avg_rl = (rl / n * 100.0) if n else 0.0
    print(f"Results on {n} items → Acc: {acc:.1f}% | R-1: {avg_r1:.1f}% | R-2: {avg_r2:.1f}% | R-L: {avg_rl:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct","interactive"], required=True)
    ap.add_argument("--suite", choices=["fanoutqa","frames","assistantbench"], required=True)
    ap.add_argument("--data", required=False, help="Path to dataset (json/jsonl)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    # Load dataset
    if args.suite == "fanoutqa":
        items = load_fanoutqa(args.data, limit=args.limit)
    elif args.suite == "frames":
        items = load_frames(args.data, limit=args.limit)
    else:
        items = load_assistantbench(args.data, limit=args.limit)

    if args.mode == "direct":
        out = run_direct(items, k=args.k)
    else:
        out = run_interactive(items, k=args.k)

    # Print and report
    for r in out:
        print(json.dumps(r, ensure_ascii=False))
    report(out)

if __name__ == "__main__":
    main()