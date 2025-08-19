# experiments/policies.py
import os, json, requests
from typing import List, Dict

WIKI_API = "https://en.wikipedia.org/w/api.php"
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/search"

# ——— Wikipedia‑only search/fetch ———

def wiki_search(query: str, k: int = 5) -> List[Dict]:
    params = {
        "action": "query", "list": "search", "srsearch": query,
        "format": "json", "srlimit": k
    }
    r = requests.get(WIKI_API, params=params, timeout=10)
    data = r.json()
    hits = data.get("query", {}).get("search", [])
    return [{"title": h["title"], "snippet": h.get("snippet", ""), "link": f"https://en.wikipedia.org/wiki/{h['title'].replace(' ', '_')}"} for h in hits]


def wiki_fetch(title: str) -> str:
    # REST plain text endpoint is concise and fast
    url = f"https://en.wikipedia.org/api/rest_v1/page/plain/{title.replace(' ', '_')}"
    try:
        r = requests.get(url, timeout=10)
        return r.text
    except Exception:
        return ""

# ——— Domain‑allowlisted web search (uses Serper if available) ———

ALLOW_DOMAINS = ["wikipedia.org"]


def restricted_search(query: str, k: int = 5):
    if os.getenv("WIKI_ONLY", "0") == "1":
        return wiki_search(query, k)
    if not SERPER_API_KEY:
        return wiki_search(query, k)
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    q = query + " site:" + " OR site:".join(ALLOW_DOMAINS)
    try:
        r = requests.post(SERPER_URL, headers=headers, data=json.dumps({"q": q, "num": k}), timeout=10)
        r.raise_for_status()
        return r.json().get("organic", [])
    except Exception:
        return wiki_search(query, k)