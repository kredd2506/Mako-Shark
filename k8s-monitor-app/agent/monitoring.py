import requests
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from tabulate import tabulate

# Initialize Kubernetes client
try:
    config.load_incluster_config()
except:
    try:
        config.load_kube_config()
    except:
        pass

v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()
batch_v1 = client.BatchV1Api()
networking_v1 = client.NetworkingV1Api()

def describe_pods(namespace="gsoc"):
    """Describe pods and print only fields useful for Prometheus metric queries."""
    try:
        pods = v1.list_namespaced_pod(namespace=namespace) if namespace else v1.list_pod_for_all_namespaces()
        rows = []
        for pod in pods.items:
            pod_name = pod.metadata.name
            ns = pod.metadata.namespace
            pod_ip = pod.status.pod_ip
            node = pod.spec.node_name
            container_names = [c.name for c in pod.spec.containers]
            container = ", ".join(container_names)
            rows.append([pod_name, ns, pod_ip, node, container])
        headers = ["Pod", "Namespace", "Pod IP", "Node", "Container"]
        return tabulate(rows, headers=headers, tablefmt="fancy_grid")
    except ApiException as e:
        return f"❌ Error fetching pods: {e}"

def namespace_gpu_utilization(prom_url="https://prometheus.nrp-nautilus.io", threshold=0):
    """Display average GPU utilization per namespace using PromQL."""
    query = 'avg by (namespace) (DCGM_FI_DEV_GPU_UTIL)'
    url = f"{prom_url}/api/v1/query"
    try:
        response = requests.get(url, params={"query": query}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "success":
            return "❌ Prometheus query failed."
        results = data["data"]["result"]
        if not results:
            return "✅ Query successful, but no GPU usage data returned."
        rows = []
        for r in results:
            ns = r["metric"].get("namespace", "unknown")
            util = float(r["value"][1])
            if util >= threshold:
                status = (
                    "🟢 Low" if util < 40 else
                    "🟡 Moderate" if util < 70 else
                    "🔴 High"
                )
                rows.append([ns, f"{util:.2f}%", status])
        headers = ["Namespace", "Avg GPU Utilization", "Status"]
        return tabulate(rows, headers=headers, tablefmt="fancy_grid")
    except Exception as e:
        return f"❌ Error querying Prometheus: {e}"

def fetch_dcgm_gpu_util_data(prom_url="https://prometheus.nrp-nautilus.io"):
    """Fetch rich GPU utilization data from Prometheus using DCGM_FI_DEV_GPU_UTIL."""
    query = 'DCGM_FI_DEV_GPU_UTIL'
    url = f"{prom_url}/api/v1/query"
    try:
        response = requests.get(url, params={"query": query}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "success":
            return []
        results = data["data"]["result"]
        if not results:
            return []
        enriched = []
        for r in results:
            m = r["metric"]
            val = float(r["value"][1])
            enriched.append({
                "hostname": m.get("Hostname", "unknown"),
                "ip_port": m.get("instance", "unknown"),
                "gpu_id": m.get("gpu", "N/A"),
                "device": m.get("device", "N/A"),
                "uuid": m.get("UUID", "N/A"),
                "model": m.get("modelName", "unknown"),
                "namespace": m.get("namespace", "N/A"),
                "pod": m.get("pod", "N/A"),
                "utilization": val
            })
        return enriched
    except Exception as e:
        return []

def analyze_dcgm_gpu_data(data):
    """Analyze DCGM GPU data with statistics and top utilization."""
    if not data:
        return "No data to analyze."
    total = len(data)
    avg_util = sum(d["utilization"] for d in data) / total
    maxed = [d for d in data if d["utilization"] >= 99.0]
    idle = [d for d in data if d["utilization"] < 1.0]
    available = [d for d in data if d["utilization"] < 100.0]
    unique_hosts = set(d["hostname"] for d in data)
    unique_models = set(d["model"] for d in data)
    
    result = f"""🔍 Total GPUs: {total}
📊 Average Utilization: {avg_util:.2f}%
🔴 Fully Utilized GPUs (>=99%): {len(maxed)}
🟢 Idle GPUs (<1%): {len(idle)}
💻 Unique Host Machines: {len(unique_hosts)}
🧠 Unique GPU Models: {len(unique_models)}
🧮 GPUs Available (<100%): {len(available)}\n"""
    
    result += "📈 Top 10 GPUs by Utilization:\n"
    top = sorted(data, key=lambda x: x["utilization"], reverse=True)[:10]
    rows = [[d["hostname"], d["gpu_id"], d["model"], f"{d['utilization']:.2f}%", d["namespace"], d["pod"]] for d in top]
    result += tabulate(rows, headers=["Host", "GPU", "Model", "Utilization", "Namespace", "Pod"], tablefmt="github")
    return result