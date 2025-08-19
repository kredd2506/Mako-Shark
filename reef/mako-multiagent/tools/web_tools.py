# tools/web_tools.py
import os
import json
import uuid
import requests
from dataclasses import dataclass, field
from datetime import datetime
from bs4 import BeautifulSoup

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SEARCH_URL = "https://google.serper.dev/search"
HDRS = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}

@dataclass
class Query:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    ts: datetime = field(default_factory=datetime.now)


def search(query: str, k: int = 5):
    if not SERPER_API_KEY:
        return []
    payload = json.dumps({"q": query, "num": k})
    try:
        r = requests.post(SEARCH_URL, headers=HDRS, data=payload, timeout=10)
        r.raise_for_status()
        return r.json().get('organic', [])
    except Exception:
        return []


def fetch_text(url: str):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        for el in soup(["script", "style", "nav", "footer"]):
            el.decompose()
        content = soup.find('main') or soup.find('article') or soup
        return content.get_text(" ", strip=True)[:2000]
    except Exception:
        return None


def ask(raw: str):
    q = raw.strip() or "Kubernetes"
    results = search(f"{q} site:kubernetes.io OR site:github.com/kubernetes", k=5)
    lines = []
    for res in results[:5]:
        url = res.get("link")
        title = res.get("title")
        text = fetch_text(url) or "(no extract)"
        lines.append(f"- {title}\n  {url}\n  {text[:160]}...")
    return "\n".join(lines) or "No results"

ACTIONS = {"web_search": ask}