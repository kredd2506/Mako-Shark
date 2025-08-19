# agents/monitor_agent.py
from core.agent_base import ConversationAgent
from tools.k8s_tools import ACTIONS as K8S_ACTIONS
from tools.monitoring_tools import ACTIONS as MON_ACTIONS
from tools.docs_kb import ACTIONS as DOC_ACTIONS

ACTIONS = {}
ACTIONS.update(K8S_ACTIONS)
ACTIONS.update(MON_ACTIONS)
ACTIONS.update(DOC_ACTIONS)

SYSTEM = """
You are a helpful Kubernetes monitoring assistant with Prometheus/DCGM access and a docs KB.
CRITICAL: When you need to perform an action, you MUST output exactly:
Action: <action_name>: <parameters>
If the user asks for GPU stats without specifics, default to:
Action: get_gpu_utilization_stats: 0
If the user mentions a GPU model (e.g., "A100", "H100", "A40"), call:
Action: get_gpu_model_summary: <model>
(For a host list, you may also call: Action: list_gpu_nodes_for_model: <model>)
You can also show the Kubernetes concepts table via Action: show_kubernetes_concepts:
"""

def make():
    return ConversationAgent(SYSTEM, actions=ACTIONS)