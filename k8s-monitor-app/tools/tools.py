import sys
import json
from io import StringIO
from langchain_core.tools import tool
from agent.monitoring import describe_pods, namespace_gpu_utilization, fetch_dcgm_gpu_util_data, analyze_dcgm_gpu_data
from tabulate import tabulate

# Helper to capture and truncate printed output
def capture_stdout_truncated(func, max_length=2000, *args, **kwargs):
    """Capture stdout and truncate if too long to prevent LLM loops"""
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    
    output = mystdout.getvalue()
    if len(output) > max_length:
        output = output[:max_length] + f"\n\n... [Output truncated - showing first {max_length} characters]"
    return output

# Documentation search tool
@tool
def search_documentation(query: str) -> str:
    """Search the NRP.ai documentation for relevant information."""
    # This will be set in the main app
    from app import doc_kb
    
    if doc_kb.embeddings is None:
        return "❌ Knowledge base not loaded. Cannot search documentation."
    
    results = doc_kb.search(query, top_k=3)
    if not results:
        return "❌ No relevant documentation found."
    
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"Result {i}:")
        output.append(f"Title: {result['title']}")
        output.append(f"URL: {result['url']}")
        output.append(f"Content: {result['text'][:200]}...")
        output.append("")  # Empty line
    
    return "\n".join(output)

# GPU monitoring tools
@tool
def describe_pods_tool(namespace: str = "gsoc") -> str:
    """Describe pods in a given Kubernetes namespace. Defaults to 'gsoc'."""
    return capture_stdout_truncated(describe_pods, 1500, namespace=namespace)

@tool
def namespace_gpu_util_tool(threshold: float = 0.0) -> str:
    """Get average GPU utilization per namespace with optional threshold filter."""
    return capture_stdout_truncated(namespace_gpu_utilization, 1500, threshold=threshold)

@tool
def dcgm_gpu_inspect_tool(threshold: float = 0.0) -> str:
    """Inspect raw GPU usage with model name, host, pod, and utilization. Shows top 10 GPUs above threshold."""
    data = fetch_dcgm_gpu_util_data()
    if not data:
        return "⚠️ No GPU data available."
    filtered = [d for d in data if d["utilization"] >= threshold]
    if not filtered:
        return f"✅ No GPUs over {threshold}% utilization."
    top = sorted(filtered, key=lambda x: x["utilization"], reverse=True)[:10]
    rows = [
        [d["hostname"][:20], d["gpu_id"], d["model"][:25], f"{d['utilization']:.2f}%", d["namespace"], d["pod"][:20]]
        for d in top
    ]
    result = tabulate(rows, headers=["Host", "GPU", "Model", "Util%", "Namespace", "Pod"], tablefmt="grid")
    result += f"\n\nShowing top 10 of {len(filtered)} GPUs above {threshold}% threshold."
    return result

@tool
def calculate_dcgm_gpu_stats(threshold: float = 0.0) -> str:
    """Analyze GPU utilization across nodes and return statistical breakdown."""
    data = fetch_dcgm_gpu_util_data()
    if not data:
        return "⚠️ No GPU data available."
    filtered = [d for d in data if d["utilization"] >= threshold]
    total = len(filtered)
    if total == 0:
        return f"✅ No GPUs over the threshold of {threshold}% utilization."
    avg_util = sum(d["utilization"] for d in filtered) / total
    maxed = [d for d in filtered if d["utilization"] >= 99.0]
    idle = [d for d in filtered if d["utilization"] < 1.0]
    moderate = [d for d in filtered if 1.0 <= d["utilization"] < 70.0]
    available = [d for d in filtered if d["utilization"] < 100.0]
    unique_models = set(d["model"] for d in filtered)
    unique_hosts = set(d["hostname"] for d in filtered)
    return f"""📊 GPU Utilization Stats (threshold: {threshold}%):
🔍 Total GPUs: {total}
📈 Average Utilization: {avg_util:.2f}%
🔴 Fully Utilized (>=99%): {len(maxed)}
🟢 Idle (<1%): {len(idle)}
⚙️ Moderate (1-70%): {len(moderate)}
💻 Unique Hosts: {len(unique_hosts)}
🧠 Unique Models: {len(unique_models)}
🧮 Available (<100%): {len(available)}"""