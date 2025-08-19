# %%
# %% [markdown]
# # Documentation Search Integration
# %% [markdown]
# This section provides documentation search capabilities
# %%
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
# %%
os.environ["NRP_API_KEY"] = "NRP-API-key-here"
# %%
import re

# %%
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
config.load_incluster_config()
v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()
batch_v1 = client.BatchV1Api()
networking_v1 = client.NetworkingV1Api()
# %%
# %% [markdown]
# Implementing a simple react pattern
# %%
from openai import OpenAI
client = OpenAI(
    # This is the default and can be omitted
    api_key = os.environ.get("NRP_API_KEY"),
    base_url = "https://llm.nrp-nautilus.io/"
)
completion = client.chat.completions.create(
    model="gemma3",
    messages=[
        {"role": "system", "content": "You are a helpful Kubernetes assistant."},
    ],
)
# %%

# %% [markdown]
# ## Documentation Knowledge Base Class
# %%
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
                response = self.session.post(self.embedding_endpoint, json=data, timeout=30)
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
            response = self.session.post(self.rerank_endpoint, json=data, timeout=30)
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
        
        # Apply reranking if requested
        if use_reranking and len(results) > 0:
            results = self.rerank_results(query, results, top_k)
            
        return results
    
    def load_knowledge_base(self, filepath):
        """Load the knowledge base from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.documents = data['documents']
        self.embeddings = np.array(data['embeddings'])
        self.metadata = data['metadata']
        
        print(f"Knowledge base loaded from {filepath} with {len(self.documents)} documents")
# %% [markdown]
# ## Initialize Documentation Knowledge Base
# %%
# Initialize the knowledge base
doc_kb = DocumentationKnowledgeBase()
# Try to load the knowledge base from file
kb_file = "nrp_expert_docs_kb.json"
if os.path.exists(kb_file):
    doc_kb.load_knowledge_base(kb_file)
else:
    print(f"❌ Knowledge base file {kb_file} not found. Please provide a pre-built knowledge base file.")
# %% [markdown]
# ## Documentation Search Function
# %%
def search_documentation(query):
    """
    Search the NRP.ai documentation for the given query.
    Returns a formatted string with the top results.
    """
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

# %%
# %%
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
                        temperature=0,
                        messages=self.messages)
        return completion.choices[0].message.content
# %%
prompt = """
You are a Kubernetes assistant with access to NRP.ai documentation. You operate in a loop of:
Thought → Action → PAUSE → Observation
At the end of this loop, you output a final Answer.
---
**Instructions:**
- Use **Thought** to explain your reasoning based on the user's request.
- Use **Action** to call one of the tools listed below. Each Action must be followed by **PAUSE** so the system can run the tool.
- The result of the action will be passed back to you as an **Observation**.
- After processing the Observation, continue the loop.
- Stop when you have gathered enough information, and provide an **Answer**.
---
**Namespace Rule:**
- Always check if the namespace is set before performing any namespaced action.
- If it is not set, ask: *"Which namespace should I use?"*
- Then call: `set_namespace: <namespace>`
---
**When to Use `describe_*` Tools:**
- If the user mentions a specific name (e.g., "ubuntu"), check for matching resources using `list_*` tools.
- If a match is found, use the appropriate `describe_*` tool for detailed information.
- If multiple resources match the name, describe each one.
- Only use `describe_*` if you're confident about the target resource name.
---
**Documentation Search:**
- When users ask about Kubernetes concepts, GPU pods, or cloud computing that isn't directly about cluster resources, use `search_documentation`.
- This will search the NRP.ai documentation for relevant information.
---
**Available Actions:**
set_namespace:
e.g. set_namespace: kube-system
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
Each of the above lists the corresponding resources.
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
Each of the above describes the specified resource.
search_documentation:
e.g. search_documentation: How to set up GPU pods in Kubernetes?
Searches the NRP.ai documentation for relevant information.
---
**Example 1:**
Question: What pods are running?
Thought: I need to check if the namespace is already set. Since it isn't, I will ask the user.
Action: set_namespace: kube-system
PAUSE
(Observation: ✅ Namespace set to 'kube-system')
Thought: Now I can list the pods in the kube-system namespace.
Action: list_pods:
PAUSE
(Observation: ['coredns-abc123', 'kube-proxy-xyz789'])
Answer: The pods currently running in kube-system are: coredns-abc123, kube-proxy-xyz789.
---
**Example 2:**
Question: What is happening with ubuntu?
Thought: The user asked about something named 'ubuntu'. I will first list pods to see if any match.
Action: list_pods:
PAUSE
(Observation: ['ubuntu-runner-xyz', 'nginx'])
Thought: A pod named 'ubuntu-runner-xyz' matches. I will describe it.
Action: describe_pod: ubuntu-runner-xyz
PAUSE
(Observation: 📋 Pod 'ubuntu-runner-xyz' phase: Running, node: node-123)
Answer: The pod 'ubuntu-runner-xyz' is currently running on node node-123.
---
**Example 3:**
Question: How do I set up GPU pods in Kubernetes?
Thought: The user is asking about setting up GPU pods. This is a configuration question that might be answered in the documentation. I will search the documentation.
Action: search_documentation: How to set up GPU pods in Kubernetes?
PAUSE
(Observation: Result 1:
Title: GPU Support in Kubernetes
URL: https://nrp.ai/documentation/userdocs/gpu-support
Content: Kubernetes provides support for GPUs through device plugins. To use GPUs in your pods...
)
Thought: The documentation search returned relevant information about GPU support in Kubernetes. I can now provide an answer based on this.
Answer: According to the NRP.ai documentation, Kubernetes provides support for GPUs through device plugins. You can set up GPU pods by configuring your pods to request GPU resources. For detailed steps, see: https://nrp.ai/documentation/userdocs/gpu-support
---
"""
# %%
# %%
CURRENT_NAMESPACE = None
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

# %%
import re
from kubernetes.client.rest import ApiException
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

# %%
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
    "search_documentation": search_documentation,
}
# %%
# %% [markdown]
# ## Add Loop
# %%
action_re = re.compile(r'^Action: (\w+):(.*)$')
# %%
# %%
def query(question, max_turns=15):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        print(f"\n--- Turn {i+1} ---")
        i += 1
        print("Prompt to bot:", next_prompt)
        result = bot(next_prompt)
        print("Bot response:\n", result)
        actions = [
            action_re.match(a)
            for a in result.split('\n')
            if action_re.match(a)
        ]
        if actions:
            for action_match in actions:
                action, action_input = action_match.groups()
                if action not in known_actions:
                    raise Exception(f"Unknown action: {action}: {action_input}")
                print(f" -- Running action '{action}' with input '{action_input}'")
                observation = known_actions[action](action_input)
                print("Observation:", observation)
                next_prompt = f"Observation: {observation}"
        else:
            print("No more actions. Halting.")
            return
# %%
# %% [markdown]
# ## Test Documentation Search
# %%
# Create a new agent with the updated prompt
abot = Agent(prompt)
# Test the documentation search
question = "How do I configure persistent storage in Kubernetes?"
result = abot(question)
print(result)
# %%
# %%
question = """I want to see everything related to the name ubuntu in gsoc namespace and explain them"""
query(question)