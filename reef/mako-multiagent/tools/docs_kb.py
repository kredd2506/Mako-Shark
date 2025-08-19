# tools/docs_kb.py
import os
import json
import time
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

NRP_API_KEY = os.environ.get("NRP_API_KEY", "")
NRP_BASE_URL = os.environ.get("NRP_BASE_URL", "https://llm.nrp-nautilus.io/")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embed-mistral")

class DocumentationKnowledgeBase:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.metadata = []
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {NRP_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "NRP-DocKB/1.0"
        })

    def get_embeddings(self, texts, batch_size=10):
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            r = self.session.post(f"{NRP_BASE_URL}/v1/embeddings", json={"model": EMBED_MODEL, "input": batch}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                all_vecs.extend([d['embedding'] for d in data['data']])
            else:
                all_vecs.extend([[0.0]*768]*len(batch))
            time.sleep(0.5)
        return all_vecs

    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.documents = data['documents']
        self.embeddings = np.array(data['embeddings'])
        self.metadata = data['metadata']
        return f"KB loaded: {len(self.documents)} docs"

    def search(self, query: str, top_k=3):
        if self.embeddings is None:
            return []
        qv = np.array(self.get_embeddings([query])[0]).reshape(1, -1)
        sims = cosine_similarity(qv, self.embeddings).flatten()
        idxs = sims.argsort()[-top_k:][::-1]
        results = []
        for idx in idxs:
            results.append({
                'text': self.documents[idx]['text'],
                'url': self.metadata[idx]['url'],
                'title': self.metadata[idx]['title'],
                'score': float(sims[idx])
            })
        return results

KB = DocumentationKnowledgeBase()


def search_documentation(raw: str):
    # support optional ", true" to skip rerank like your a2
    query = raw.split(",")[0].strip()
    if not KB.embeddings is None:
        hits = KB.search(query, top_k=3)
        if not hits:
            return "❌ No relevant documentation found."
        lines = []
        for i, h in enumerate(hits, 1):
            lines.append(f"Result {i}:\nTitle: {h['title']}\nURL: {h['url']}\nContent: {h['text'][:220]}...")
        return "\n\n".join(lines)
    return "❌ Knowledge base not loaded."

ACTIONS = {"search_documentation": search_documentation}