# app.py
from dotenv import load_dotenv
load_dotenv(override=True)

import os
from orchestrator.router import route
from agents import k8s_react_agent, monitor_agent, web_nav_agent
from tools.docs_kb import KB

BANNER = """
🚀 Mako Multi‑Agent CLI
Type 'exit' to quit. Examples:
  - list pods
  - describe_pod: mypod-123
  - gpu stats
  - show kubernetes concepts
  - search: how to deploy statefulset
"""

AGENTS = {
    'K8S': k8s_react_agent.make(),
    'MONITOR': monitor_agent.make(),
    'WEB': web_nav_agent.make(),
}


def ensure_env():
    if not os.environ.get("NRP_API_KEY"):
        print("⚠️  NRP_API_KEY not set; LLM calls will fail.")


def maybe_load_kb():
    kb_path = os.environ.get("DOC_KB_PATH", "nrp_expert_docs_kb.json")
    if os.path.exists(kb_path):
        print(KB.load(kb_path))
    else:
        print(f"(Doc KB not found at {kb_path}; doc search disabled)")


def main():
    ensure_env()
    maybe_load_kb()
    print(BANNER)
    while True:
        try:
            user = input("\n> ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                print("Bye!")
                return
            dest = route(user)
            agent = AGENTS.get(dest)
            if not agent:
                print("Router failed; defaulting to K8S agent")
                agent = AGENTS['K8S']
            out = agent.run(user)
            print("\n" + str(out))
        except KeyboardInterrupt:
            print("\nBye!")
            return
        except Exception as e:
            print(f"❌ {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()