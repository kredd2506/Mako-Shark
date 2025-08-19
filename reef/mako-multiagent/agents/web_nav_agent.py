# agents/web_nav_agent.py
from core.agent_base import ConversationAgent
from tools.web_tools import ACTIONS as WEB_ACTIONS

SYSTEM = """
You are a web navigator/extractor for Kubernetes topics. Use web_search to gather sources and present a grounded summary.
Always call: Action: web_search: <query>
Then summarize concisely with bullet points and include the URLs you saw.
"""

def make():
    return ConversationAgent(SYSTEM, actions=WEB_ACTIONS)