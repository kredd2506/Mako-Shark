import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tabulate import tabulate
from openai import OpenAI
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
import sys

# Initialize environment variables
os.environ["NRP_API_KEY"] = "NRP-API-key-here"

# Initialize Kubernetes client
try:
    # Try to load in-cluster config first
    config.load_incluster_config()
    print("✅ Using in-cluster Kubernetes configuration")
except config.ConfigException:
    try:
        # Fall back to kubeconfig file
        config.load_kube_config()
        print("✅ Using local kubeconfig file")
    except:
        print("❌ Failed to initialize Kubernetes client: No valid configuration found")
        print("Please ensure you're either running inside a cluster or have a valid kubeconfig file")
        sys.exit(1)

v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()
batch_v1 = client.BatchV1Api()
networking_v1 = client.NetworkingV1Api()
print("✅ Kubernetes client initialized successfully")

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("NRP_API_KEY"),
    base_url="https://llm.nrp-nautilus.io/"
)

# Global namespace variable
CURRENT_NAMESPACE = None

class DocumentationKnowledgeBase:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.metadata = []
        self.api_key = os.environ.get("NRP_API_KEY", "NRP-API-key-here")
        self.base_url = "https://llm.nrp-nautilus.io/"
        self.embedding_endpoint = f"{self.base_url}/v1/embeddings"
        self.rerank_endpoint = f"{self.base_url}/v1/rerank"
        
        # Create a robust session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "User-Agent": "NRP-Documentation-Crawler/1.0",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
    def get_embeddings(self, texts, batch_size=10):
        """Get embeddings from the NRP API with batching"""
        all_embeddings = []
        
        # Process texts in batches to avoid overwhelming the API
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            data = {
                "model": "embed-mistral",
                "input": batch
            }
            
            try:
                response = self.session.post(self.embedding_endpoint, json=data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    all_embeddings.extend([item['embedding'] for item in result['data']])
                    print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                else:
                    print(f"Error getting embeddings: {response.status_code} - {response.text}")
                    # Add zero embeddings as fallback
                    all_embeddings.extend([[0.0] * 768] * len(batch))
            except Exception as e:
                print(f"Exception when getting embeddings: {e}")
                # Add zero embeddings as fallback
                all_embeddings.extend([[0.0] * 768] * len(batch))
                
            # Add delay between batches
            time.sleep(1)
            
        return all_embeddings
    
    def rerank_results(self, query, documents, top_k=5):
        """Rerank search results using the NRP API"""
        # Prepare documents for reranking
        docs_for_rerank = [{"text": doc['text']} for doc in documents]
        
        data = {
            "model": "gemma3",
            "query": query,
            "documents": docs_for_rerank,
            "top_n": top_k
        }
        
        try:
            # Reduced timeout to prevent hanging
            response = self.session.post(self.rerank_endpoint, json=data, timeout=5)
            if response.status_code == 200:
                result = response.json()
                # Get indices of top results
                top_indices = [item['index'] for item in result['results']]
                return [documents[i] for i in top_indices]
            else:
                print(f"Error reranking results: {response.status_code} - {response.text}")
                return documents[:top_k]  # Fallback to original order
        except Exception as e:
            print(f"Exception when reranking: {e}")
            return documents[:top_k]  # Fallback to original order
    
    def search(self, query, top_k=5, use_reranking=True):
        """Search the knowledge base"""
        if self.embeddings is None:
            print("Knowledge base not loaded. Please load it first.")
            return []
            
        # Get query embedding
        query_embedding = self.get_embeddings([query])
        if query_embedding is None:
            print("Failed to generate query embedding")
            return []
            
        query_embedding = np.array(query_embedding[0]).reshape(1, -1)
        
        # Calculate similarity
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx]['text'],
                'url': self.metadata[idx]['url'],
                'title': self.metadata[idx]['title'],
                'score': float(similarities[idx])
            })
        
        # Apply reranking if requested and not disabled
        if use_reranking and len(results) > 0:
            try:
                results = self.rerank_results(query, results, top_k)
            except KeyboardInterrupt:
                print("Reranking interrupted. Using original results.")
                # Return original results if reranking is interrupted
                pass
            
        return results
    
    def load_knowledge_base(self, filepath):
        """Load the knowledge base from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.documents = data['documents']
        self.embeddings = np.array(data['embeddings'])
        self.metadata = data['metadata']
        
        print(f"Knowledge base loaded from {filepath} with {len(self.documents)} documents")

# Initialize the knowledge base
doc_kb = DocumentationKnowledgeBase()
# Try to load the knowledge base from file
kb_file = "nrp_expert_docs_kb.json"
if os.path.exists(kb_file):
    doc_kb.load_knowledge_base(kb_file)
else:
    print(f"❌ Knowledge base file {kb_file} not found. Documentation search will not be available.")

# Prometheus Monitoring Functions
def describe_pods(namespace=None):
    """
    Describe pods and print only fields useful for Prometheus metric queries.
    If namespace is None, use current namespace. If namespace is "all", list all pods.
    """
    try:
        if namespace == "all":
            pods = v1.list_pod_for_all_namespaces()
        elif namespace is None:
            namespace = get_namespace()
            pods = v1.list_namespaced_pod(namespace=namespace)
        else:
            pods = v1.list_namespaced_pod(namespace=namespace)
            
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

def namespace_gpu_utilization(threshold=0):
    """
    Display average GPU utilization per namespace using PromQL.
    Args:
        threshold (float): Minimum % utilization to show (filtering).
    """
    query = 'avg by (namespace) (DCGM_FI_DEV_GPU_UTIL)'
    url = f"https://prometheus.nrp-nautilus.io/api/v1/query"
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

def fetch_dcgm_gpu_util_data():
    """
    Fetch rich GPU utilization data from Prometheus using DCGM_FI_DEV_GPU_UTIL.
    
    Returns:
        list of dicts with context: [{hostname, gpu_id, model, namespace, pod, utilization, ...}]
    """
    query = 'DCGM_FI_DEV_GPU_UTIL'
    url = f"https://prometheus.nrp-nautilus.io/api/v1/query"
    try:
        response = requests.get(url, params={"query": query}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "success":
            print("❌ Prometheus query failed.")
            return []
        results = data["data"]["result"]
        if not results:
            print("✅ Query successful, but no GPU data returned.")
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
        print(f"❌ Error querying Prometheus: {e}")
        return []

def get_gpu_utilization_details(top_n=10, threshold=0):
    """
    Get detailed GPU utilization data.
    Args:
        top_n: Number of top GPUs to display
        threshold: Minimum utilization threshold to include
    """
    data = fetch_dcgm_gpu_util_data()
    if not data:
        return "⚠️ No GPU data available."
    filtered = [d for d in data if d["utilization"] >= threshold]
    if not filtered:
        return f"✅ No GPUs over {threshold}% utilization."
    top = sorted(filtered, key=lambda x: x["utilization"], reverse=True)[:top_n]
    rows = [
        [d["hostname"][:20], d["gpu_id"], d["model"][:25], f"{d['utilization']:.2f}%", d["namespace"], d["pod"][:20]]
        for d in top
    ]
    result = tabulate(rows, headers=["Host", "GPU", "Model", "Util%", "Namespace", "Pod"], tablefmt="grid")
    result += f"\n\nShowing top {len(top)} of {len(filtered)} GPUs above {threshold}% threshold."
    return result

def get_gpu_utilization_stats(threshold=0):
    """
    Get statistical breakdown of GPU utilization.
    Args:
        threshold: Minimum utilization threshold to include
    """
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

def search_documentation(query, skip_rerank=False):
    """
    Search the NRP.ai documentation for the given query.
    Returns a formatted string with the top results.
    Args:
        query: The search query
        skip_rerank: If True, skip reranking to avoid potential timeouts
    """
    if doc_kb.embeddings is None:
        return "❌ Knowledge base not loaded. Cannot search documentation."
    
    try:
        results = doc_kb.search(query, top_k=3, use_reranking=not skip_rerank)
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
    except KeyboardInterrupt:
        return "❌ Search was interrupted. Please try again."
    except Exception as e:
        return f"❌ Error during search: {str(e)}"

# Kubernetes Concepts Table Function
def show_kubernetes_concepts(_=None):
    """
    Display a table of key Kubernetes concepts with their importance and references.
    """
    table = """
| Topic             | TL;DR                                                                          | Importance | Ref                |
| ----------------- | ------------------------------------------------------------------------------ | ---------: | ------------------ |
| Jobs              | **Run-to-completion**, not indefinite; use `restartPolicy: OnFailure` or `Never`. | High       | [Kubernetes][1]    |
| Deployments       | Long-running stateless services; rolling updates & scaling.                    | High       | [Kubernetes][2]    |
| CronJobs          | Schedule Jobs on a cron; for backups/reports.                                  | Medium     | [Kubernetes][3]    |
| StatefulSets      | Stable identity/storage for DBs; ordered, rolling updates.                     | High       | [Kubernetes][4]    |
| DaemonSets        | One Pod per node for node-level agents (logging/CNI).                          | High       | [Kubernetes][5]    |
| Probes            | Readiness gates traffic; Liveness restarts; Startup delays liveness.           | High       | [Kubernetes][6]    |
| Resources & QoS   | Always set requests/limits; QoS affects eviction priority.                     | High       | [Kubernetes][7]    |
| Services          | Types: ClusterIP/NodePort/LoadBalancer (expose Pods).                          | High       | [Kubernetes][8]    |
| Ingress           | L7 HTTP(S) routing; **requires** an ingress controller.                        | High       | [Kubernetes][9]    |
| NetworkPolicy     | Control Pod-to-Pod traffic; default-deny + allow needed flows.                 | High       | [Kubernetes][10]   |
| Storage           | Use PVC↔PV via StorageClass for persistence.                                   | High       | [Kubernetes][11]   |
| Scheduling        | nodeSelector/affinity; taints+tolerations; topology spread.                    | Medium     | [Kubernetes][12]   |
| Autoscaling       | HPA scales replicas by CPU/mem/metrics.                                        | Medium     | [Kubernetes][13]   |
| Disruptions       | Use **PDBs** to keep N replicas up during drains/rollouts.                     | High       | [Kubernetes][14]   |
| Security          | `securityContext` (run as non-root, drop caps); **PSA** (restricted/baseline). | High       | [Kubernetes][15]   |
| Namespaces & RBAC | Isolate resources; least-privilege with Roles/Bindings + ServiceAccounts.      | High       | [Kubernetes][16]   |
| Graceful shutdown | Use `preStop` + tune `terminationGracePeriodSeconds` (default 30s).            | Medium     | [Kubernetes][17]   |

---

**References:**
[1]: https://kubernetes.io/docs/concepts/workloads/controllers/job/
[2]: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
[3]: https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/
[4]: https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/
[5]: https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/
[6]: https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/
[7]: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
[8]: https://kubernetes.io/docs/tutorials/kubernetes-basics/expose/expose-intro/
[9]: https://kubernetes.io/docs/concepts/services-networking/ingress/
[10]: https://kubernetes.io/docs/concepts/services-networking/network-policies/
[11]: https://kubernetes.io/docs/concepts/storage/persistent-volumes/
[12]: https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/
[13]: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
[14]: https://kubernetes.io/docs/concepts/workloads/pods/disruptions/
[15]: https://kubernetes.io/docs/tasks/configure-pod-container/security-context/
[16]: https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/
[17]: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/
"""
    return table

# Agent class
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    def execute(self):
        completion = client.chat.completions.create(
                        model="gemma3", 
                        temperature=0.2,  # Increased slightly for more natural responses
                        messages=self.messages)
        return completion.choices[0].message.content

# Updated Agent prompt with clearer instructions
prompt = """
You are a helpful Kubernetes assistant with access to NRP.ai documentation and Prometheus monitoring. Your goal is to be conversational, ask clarifying questions, and guide users to the right resources.

**CRITICAL INSTRUCTION:** Whenever you need to perform an action, you MUST output it in the exact format:
Action: <action_name>: <parameters>

**Interaction Style:**
- Be conversational and helpful, not robotic
- Ask clarifying questions when needed
- Guide users through processes step by step
- Avoid repetitive answers
- Provide context and explanations

**Your Approach:**
1. When users ask for something that requires more details (like workspace changes, storage increases, etc.), first ask clarifying questions
2. Only search documentation after understanding the full context
3. Provide clear guidance on next steps
4. If a request is outside your capabilities, explain why and direct users to the appropriate resources

**Available Actions:**
set_namespace:
e.g. Action: set_namespace: kube-system
Sets the namespace for all operations.

list_pods:
list_deployments:
list_services:
list_jobs:
list_configmaps:
list_secrets:
list_pvcs:
list_replicasets:
list_statefulsets:
list_daemonsets:
list_events:
list_ingresses:
list_nodes:
Each of the above lists the corresponding resources. Example: Action: list_pods:

describe_pod:
describe_deployment:
describe_job:
describe_service:
describe_configmap:
describe_secret:
describe_pvc:
describe_replicaset:
describe_statefulset:
describe_daemonset:
describe_ingress:
describe_node:
Each of the above describes the specified resource. Example: Action: describe_pod: my-pod-name

describe_pods:
e.g. Action: describe_pods: gsoc or Action: describe_pods: all
Describes pods in a specific namespace or across all namespaces.

namespace_gpu_utilization:
e.g. Action: namespace_gpu_utilization: 10
Shows average GPU utilization per namespace with optional threshold filter.

get_gpu_utilization_details:
e.g. Action: get_gpu_utilization_details: 10, 20
Shows detailed GPU metrics for top 10 GPUs above 20% utilization.

get_gpu_utilization_stats:
e.g. Action: get_gpu_utilization_stats: 5
Shows statistical breakdown of GPU usage above 5% utilization.

search_documentation:
e.g. Action: search_documentation: How to set up GPU pods in Kubernetes?
Searches the NRP.ai documentation for relevant information.

show_kubernetes_concepts:
e.g. Action: show_kubernetes_concepts:
Displays a comprehensive table of key Kubernetes concepts with their importance and documentation references.

**Example Interaction:**
User: List all pods in the default namespace.
Assistant: I'll list all pods in the default namespace for you.
Action: set_namespace: default
Action: list_pods:

User: Show me GPU utilization across all namespaces
Assistant: I'll check the GPU utilization across all namespaces for you.
Action: namespace_gpu_utilization: 0

User: could I get my Coder workspace (named "hls4ml-tutorial") increased to 50 GB please? My email is apk66@drexel.edu
Assistant: I'd be happy to help you with increasing your Coder workspace storage. To better assist you, could you please provide a few more details?
1. Are you referring to the Coder workspace on the NRP platform?
2. Have you already checked the current storage allocation for your workspace?
3. Is this for a specific project or workload that requires the additional storage?
Once I have these details, I can guide you to the right process for requesting a storage increase.

User: Show me Kubernetes concepts reference
Assistant: I'll display a comprehensive table of key Kubernetes concepts with their importance and documentation references.
Action: show_kubernetes_concepts:

**IMPORTANT:** When a user asks for GPU stats without specifying details, provide a statistical overview by default using get_gpu_utilization_stats with threshold 0.
"""

# Namespace functions
def set_namespace(ns):
    """
    Set the global namespace for all Kubernetes operations.
    """
    global CURRENT_NAMESPACE
    CURRENT_NAMESPACE = ns.strip()
    return f"✅ Namespace set to '{CURRENT_NAMESPACE}'"

def get_namespace():
    """
    Retrieve the currently set namespace.
    Raises an error if namespace is not set.
    """
    if CURRENT_NAMESPACE is None:
        raise ValueError("❌ Namespace not set. Use `set_namespace` first.")
    return CURRENT_NAMESPACE

# Kubernetes resource functions
def validate_k8s_name(name):
    """
    Validate that the name follows Kubernetes RFC1123 naming convention.
    """
    pattern = r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$'
    if not re.match(pattern, name):
        raise ValueError(f"❌ Invalid Kubernetes resource name: '{name}'. Must match RFC1123 format.")
    return name

# ---------- LIST FUNCTIONS ----------
def list_pods(_=None):
    namespace = get_namespace()
    pods = v1.list_namespaced_pod(namespace=namespace)
    return [pod.metadata.name for pod in pods.items]

def list_deployments(_=None):
    namespace = get_namespace()
    deployments = apps_v1.list_namespaced_deployment(namespace=namespace)
    return [d.metadata.name for d in deployments.items]

def list_services(_=None):
    namespace = get_namespace()
    services = v1.list_namespaced_service(namespace=namespace)
    return [s.metadata.name for s in services.items]

def list_jobs(_=None):
    namespace = get_namespace()
    jobs = batch_v1.list_namespaced_job(namespace=namespace)
    return [j.metadata.name for j in jobs.items]

def list_configmaps(_=None):
    namespace = get_namespace()
    cms = v1.list_namespaced_config_map(namespace=namespace)
    return [cm.metadata.name for cm in cms.items]

def list_secrets(_=None):
    namespace = get_namespace()
    secrets = v1.list_namespaced_secret(namespace=namespace)
    return [s.metadata.name for s in secrets.items]

def list_pvcs(_=None):
    namespace = get_namespace()
    pvcs = v1.list_namespaced_persistent_volume_claim(namespace=namespace)
    return [p.metadata.name for p in pvcs.items]

def list_replicasets(_=None):
    namespace = get_namespace()
    rsets = apps_v1.list_namespaced_replica_set(namespace=namespace)
    return [r.metadata.name for r in rsets.items]

def list_statefulsets(_=None):
    namespace = get_namespace()
    ssets = apps_v1.list_namespaced_stateful_set(namespace=namespace)
    return [s.metadata.name for s in ssets.items]

def list_daemonsets(_=None):
    namespace = get_namespace()
    dsets = apps_v1.list_namespaced_daemon_set(namespace=namespace)
    return [d.metadata.name for d in dsets.items]

def list_ingresses(_=None):
    namespace = get_namespace()
    ingresses = networking_v1.list_namespaced_ingress(namespace=namespace)
    return [i.metadata.name for i in ingresses.items]

def list_events(_=None):
    namespace = get_namespace()
    events = v1.list_namespaced_event(namespace=namespace)
    return [f"{e.last_timestamp}: {e.message}" for e in events.items]

def list_nodes(_=None):
    nodes = v1.list_node()
    return [n.metadata.name for n in nodes.items]

# ---------- DESCRIBE FUNCTIONS ----------
def describe_pod(name):
    name = name.strip()
    namespace = get_namespace()
    try:
        pod = v1.read_namespaced_pod(name=name, namespace=namespace)
        return f"📋 Pod '{name}' phase: {pod.status.phase}, node: {pod.spec.node_name}"
    except ApiException as e:
        if e.status == 404:
            return f"❌ Pod '{name}' not found in namespace '{namespace}'."
        raise  # Re-raise other exceptions

def describe_deployment(name):
    name = name.strip()
    namespace = get_namespace()
    try:
        dep = apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
        return f"📦 Deployment '{name}' has {dep.status.replicas or 0} replicas and {dep.status.ready_replicas or 0} ready."
    except ApiException as e:
        if e.status == 404:
            return f"❌ Deployment '{name}' not found in namespace '{namespace}'."
        raise

def describe_service(name):
    name = name.strip()
    namespace = get_namespace()
    try:
        svc = v1.read_namespaced_service(name=name, namespace=namespace)
        return f"🌐 Service '{name}' type: {svc.spec.type}, cluster IP: {svc.spec.cluster_ip}"
    except ApiException as e:
        if e.status == 404:
            return f"❌ Service '{name}' not found in namespace '{namespace}'."
        raise

def describe_job(name):
    name = name.strip()
    namespace = get_namespace()
    try:
        job = batch_v1.read_namespaced_job(name=name, namespace=namespace)
        return f"⚙️ Job '{name}' completions: {job.status.succeeded or 0}, active: {job.status.active or 0}"
    except ApiException as e:
        if e.status == 404:
            return f"❌ Job '{name}' not found in namespace '{namespace}'."
        raise

def describe_configmap(name):
    name = name.strip()
    namespace = get_namespace()
    try:
        cm = v1.read_namespaced_config_map(name=name, namespace=namespace)
        keys = list(cm.data.keys()) if cm.data else []
        return f"🗂️ ConfigMap '{name}' has keys: {keys}"
    except ApiException as e:
        if e.status == 404:
            return f"❌ ConfigMap '{name}' not found in namespace '{namespace}'."
        raise

def describe_secret(name):
    name = name.strip()
    namespace = get_namespace()
    try:
        sec = v1.read_namespaced_secret(name=name, namespace=namespace)
        keys = list(sec.data.keys()) if sec.data else []
        return f"🔒 Secret '{name}' contains {len(keys)} keys (values hidden)"
    except ApiException as e:
        if e.status == 404:
            return f"❌ Secret '{name}' not found in namespace '{namespace}'."
        raise

def describe_pvc(name):
    name = name.strip()
    namespace = get_namespace()
    try:
        pvc = v1.read_namespaced_persistent_volume_claim(name=name, namespace=namespace)
        return f"💾 PVC '{name}' status: {pvc.status.phase}, capacity: {pvc.status.capacity.get('storage')}"
    except ApiException as e:
        if e.status == 404:
            return f"❌ PVC '{name}' not found in namespace '{namespace}'."
        raise

def describe_replicaset(name):
    name = name.strip()
    namespace = get_namespace()
    try:
        rs = apps_v1.read_namespaced_replica_set(name=name, namespace=namespace)
        return f"📎 ReplicaSet '{name}' replicas: {rs.status.replicas}, ready: {rs.status.ready_replicas}"
    except ApiException as e:
        if e.status == 404:
            return f"❌ ReplicaSet '{name}' not found in namespace '{namespace}'."
        raise

def describe_statefulset(name):
    name = name.strip()
    namespace = get_namespace()
    try:
        ss = apps_v1.read_namespaced_stateful_set(name=name, namespace=namespace)
        return f"📘 StatefulSet '{name}' replicas: {ss.status.replicas}, ready: {ss.status.ready_replicas}"
    except ApiException as e:
        if e.status == 404:
            return f"❌ StatefulSet '{name}' not found in namespace '{namespace}'."
        raise

def describe_daemonset(name):
    name = name.strip()
    namespace = get_namespace()
    try:
        ds = apps_v1.read_namespaced_daemon_set(name=name, namespace=namespace)
        return f"🔁 DaemonSet '{name}' scheduled: {ds.status.current_number_scheduled}, ready: {ds.status.number_ready}"
    except ApiException as e:
        if e.status == 404:
            return f"❌ DaemonSet '{name}' not found in namespace '{namespace}'."
        raise

def describe_ingress(name):
    name = name.strip()
    namespace = get_namespace()
    try:
        ing = networking_v1.read_namespaced_ingress(name=name, namespace=namespace)
        hosts = [rule.host for rule in ing.spec.rules] if ing.spec.rules else []
        services = []
        for rule in ing.spec.rules or []:
            if rule.http:
                for path in rule.http.paths:
                    if path.backend and path.backend.service:
                        services.append(path.backend.service.name)
        return f"🚪 Ingress '{name}' exposes hosts: {hosts or '[]'} and forwards to services: {services or '[]'}"
    except ApiException as e:
        if e.status == 404:
            return f"❌ Ingress '{name}' not found in namespace '{namespace}'."
        raise

def describe_node(name):
    name = name.strip()
    try:
        node = v1.read_node(name=name)
        return f"🖥️ Node '{name}' labels: {node.metadata.labels}"
    except ApiException as e:
        if e.status == 404:
            return f"❌ Node '{name}' not found."
        raise

# Known actions dictionary
known_actions = {
    # Namespace control
    "set_namespace": set_namespace,
    # LIST actions
    "list_pods": list_pods,
    "list_deployments": list_deployments,
    "list_services": list_services,
    "list_jobs": list_jobs,
    "list_configmaps": list_configmaps,
    "list_secrets": list_secrets,
    "list_pvcs": list_pvcs,
    "list_replicasets": list_replicasets,
    "list_statefulsets": list_statefulsets,
    "list_daemonsets": list_daemonsets,
    "list_ingresses": list_ingresses,
    "list_events": list_events,
    "list_nodes": list_nodes,
    # DESCRIBE actions
    "describe_pod": describe_pod,
    "describe_deployment": describe_deployment,
    "describe_service": describe_service,
    "describe_job": describe_job,
    "describe_configmap": describe_configmap,
    "describe_secret": describe_secret,
    "describe_pvc": describe_pvc,
    "describe_replicaset": describe_replicaset,
    "describe_statefulset": describe_statefulset,
    "describe_daemonset": describe_daemonset,
    "describe_ingress": describe_ingress,
    "describe_node": describe_node,
    "describe_pods": describe_pods,
    # Prometheus monitoring actions
    "namespace_gpu_utilization": namespace_gpu_utilization,
    "get_gpu_utilization_details": get_gpu_utilization_details,
    "get_gpu_utilization_stats": get_gpu_utilization_stats,
    # Documentation search
    "search_documentation": search_documentation,
    # Kubernetes concepts reference
    "show_kubernetes_concepts": show_kubernetes_concepts,
}

# Query function
action_re = re.compile(r'^Action: (\w+):(.*)$', re.MULTILINE)

def process_query(bot, user_input, max_turns=15):
    """
    Process a user query with the given agent and return whether the conversation should continue.
    """
    next_prompt = user_input
    for i in range(max_turns):
        print(f"\n--- Turn {i+1} ---")
        print("Prompt to bot:", next_prompt)
        result = bot(next_prompt)
        print("Bot response:\n", result)
        
        # Find all actions in the response
        actions = []
        for line in result.split('\n'):
            match = action_re.match(line.strip())
            if match:
                actions.append(match)
        
        if actions:
            for action_match in actions:
                action, action_input = action_match.groups()
                if action not in known_actions:
                    raise Exception(f"Unknown action: {action}: {action_input}")
                print(f" -- Running action '{action}' with input '{action_input}'")
                
                # Special handling for actions with multiple parameters
                if action in ["get_gpu_utilization_details"]:
                    try:
                        # Split parameters by comma
                        params = [p.strip() for p in action_input.split(',')]
                        top_n = int(params[0]) if len(params) > 0 else 10
                        threshold = float(params[1]) if len(params) > 1 else 0
                        observation = known_actions[action](top_n=top_n, threshold=threshold)
                    except Exception as e:
                        observation = f"❌ Error parsing parameters: {str(e)}"
                elif action in ["get_gpu_utilization_stats", "namespace_gpu_utilization"]:
                    try:
                        threshold = float(action_input.strip()) if action_input.strip() else 0
                        observation = known_actions[action](threshold=threshold)
                    except Exception as e:
                        observation = f"❌ Error parsing parameter: {str(e)}"
                elif action == "search_documentation":
                    # Add option to skip reranking for search_documentation to avoid timeouts
                    try:
                        # Check if skip_rerank parameter is provided
                        if "," in action_input:
                            query_text, skip_rerank = action_input.split(",", 1)
                            skip_rerank = skip_rerank.strip().lower() == "true"
                        else:
                            query_text = action_input
                            skip_rerank = False
                        observation = known_actions[action](query_text.strip(), skip_rerank=skip_rerank)
                    except Exception as e:
                        observation = f"❌ Error during search: {str(e)}"
                else:
                    observation = known_actions[action](action_input)
                    
                print("Observation:", observation)
                next_prompt = f"Observation: {observation}"
        else:
            # No actions detected, check if the bot is asking for more information
            if "?" in result:
                # The bot is asking a question, so we need more input from the user
                # Return True to continue the conversation
                return True
            else:
                # No actions and no questions, end the conversation
                return False
    
    # If we reach max_turns, end the conversation
    return False

# Main function
def main():
    print("🚀 Kubernetes Assistant Terminal App")
    print("Type 'exit' to quit, 'help' for available commands")
    
    # Create a single agent for the entire conversation
    bot = Agent(prompt)
    
    while True:
        try:
            user_input = input("\n> ")
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
                
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("- set_namespace: <namespace> - Set the current namespace")
                print("- list_*: List Kubernetes resources (pods, deployments, etc.)")
                print("- describe_*: Describe a specific resource")
                print("- describe_pods: <namespace> - Describe pods in a namespace")
                print("- namespace_gpu_utilization: <threshold> - Show GPU utilization by namespace")
                print("- get_gpu_utilization_details: <top_n>, <threshold> - Show detailed GPU metrics")
                print("- get_gpu_utilization_stats: <threshold> - Show GPU statistics")
                print("- search_documentation: <query> - Search documentation")
                print("- show_kubernetes_concepts - Display Kubernetes concepts reference table")
                print("- Or just ask a question in natural language")
                continue
            
            # Process the query with the persistent agent
            continue_conversation = process_query(bot, user_input)
            
            # If the bot is asking for more information, continue the conversation
            while continue_conversation:
                user_input = input("\n> ")
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    return
                continue_conversation = process_query(bot, user_input)
            
        except KeyboardInterrupt:
            print("\nOperation cancelled. Type 'exit' to quit.")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()