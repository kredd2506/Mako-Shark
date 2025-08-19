# core/llm.py
import os
from openai import OpenAI

NRP_API_KEY = os.environ.get("NRP_API_KEY") or os.environ.get("NRP") or "NRP-API-key-here"
NRP_BASE_URL = os.environ.get("NRP_BASE_URL", "https://llm.nrp-nautilus.io/")
NRP_MODEL = os.environ.get("NRP_MODEL", "deepseek-r1")

if not NRP_API_KEY:
    # Don’t crash here; allow app to start and show a clear error later.
    pass

_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=NRP_API_KEY, base_url=NRP_BASE_URL)
    return _client


def chat(messages, temperature=0.2):
    client = get_client()
    resp = client.chat.completions.create(
        model=NRP_MODEL,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content