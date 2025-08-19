# agents/k8s_react_agent.py
from core.agent_base import ConversationAgent
from tools.k8s_tools import ACTIONS as K8S_ACTIONS

SYSTEM = """
You are a Kubernetes ReAct agent. You operate in a loop of:
Thought → Action → PAUSE → Observation, then a final Answer.
You are operating in the 'gsoc' namespace by default (but can change via set_namespace).
Available Actions: (from the registry) – list_*, describe_*, set_namespace, get_service_account, check_permissions.
When confident about a resource name, use describe_*; otherwise list_* first.
"""

def make():
    return ConversationAgent(SYSTEM, actions=K8S_ACTIONS)