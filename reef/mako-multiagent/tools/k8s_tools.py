# tools/k8s_tools.py
import os
import re
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

# Init K8s
try:
    config.load_incluster_config()
except config.ConfigException:
    try:
        config.load_kube_config()
    except config.ConfigException:
        pass  # defer errors until first call

v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()
batch_v1 = client.BatchV1Api()
networking_v1 = client.NetworkingV1Api()
auth_v1 = client.AuthorizationV1Api()

CURRENT_NAMESPACE = os.environ.get("DEFAULT_NS", "gsoc")

RFC1123 = re.compile(r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$')

def _ns():
    return CURRENT_NAMESPACE

def set_namespace(ns_raw: str):
    global CURRENT_NAMESPACE
    ns = (ns_raw or "").strip() or CURRENT_NAMESPACE
    CURRENT_NAMESPACE = ns
    return f"✅ Namespace set to '{CURRENT_NAMESPACE}'"

# ——— helpers ———

def _handle(e: ApiException):
    if getattr(e, 'status', None) == 403:
        return f"❌ Permission denied: {e.reason}"
    if getattr(e, 'status', None) == 404:
        return f"❌ Not found: {e.reason}"
    return f"❌ API error ({getattr(e, 'status', '?')}): {e.reason}"

# ——— LIST ———

def list_pods(_=None):
    try:
        return [p.metadata.name for p in v1.list_namespaced_pod(namespace=_ns()).items]
    except ApiException as e: return _handle(e)

def list_deployments(_=None):
    try:
        return [d.metadata.name for d in apps_v1.list_namespaced_deployment(namespace=_ns()).items]
    except ApiException as e: return _handle(e)

def list_services(_=None):
    try:
        return [s.metadata.name for s in v1.list_namespaced_service(namespace=_ns()).items]
    except ApiException as e: return _handle(e)

def list_jobs(_=None):
    try:
        return [j.metadata.name for j in batch_v1.list_namespaced_job(namespace=_ns()).items]
    except ApiException as e: return _handle(e)

def list_configmaps(_=None):
    try:
        return [cm.metadata.name for cm in v1.list_namespaced_config_map(namespace=_ns()).items]
    except ApiException as e: return _handle(e)

def list_secrets(_=None):
    try:
        return [s.metadata.name for s in v1.list_namespaced_secret(namespace=_ns()).items]
    except ApiException as e: return _handle(e)

def list_pvcs(_=None):
    try:
        return [p.metadata.name for p in v1.list_namespaced_persistent_volume_claim(namespace=_ns()).items]
    except ApiException as e: return _handle(e)

def list_replicasets(_=None):
    try:
        return [r.metadata.name for r in apps_v1.list_namespaced_replica_set(namespace=_ns()).items]
    except ApiException as e: return _handle(e)

def list_statefulsets(_=None):
    try:
        return [s.metadata.name for s in apps_v1.list_namespaced_stateful_set(namespace=_ns()).items]
    except ApiException as e: return _handle(e)

def list_daemonsets(_=None):
    try:
        return [d.metadata.name for d in apps_v1.list_namespaced_daemon_set(namespace=_ns()).items]
    except ApiException as e: return _handle(e)

def list_ingresses(_=None):
    try:
        return [i.metadata.name for i in networking_v1.list_namespaced_ingress(namespace=_ns()).items]
    except ApiException as e: return _handle(e)

def list_events(_=None):
    try:
        return [f"{e.last_timestamp}: {e.message}" for e in v1.list_namespaced_event(namespace=_ns()).items]
    except ApiException as e: return _handle(e)

def list_nodes(_=None):
    try:
        return [n.metadata.name for n in v1.list_node().items]
    except ApiException as e: return _handle(e)

# ——— DESCRIBE ———

def describe_pod(name):
    try:
        pod = v1.read_namespaced_pod(name=name.strip(), namespace=_ns())
        return f"📋 Pod '{pod.metadata.name}' phase: {pod.status.phase}, node: {pod.spec.node_name}"
    except ApiException as e: return _handle(e)

def describe_deployment(name):
    try:
        d = apps_v1.read_namespaced_deployment(name=name.strip(), namespace=_ns())
        return f"📦 Deployment '{d.metadata.name}' replicas: {d.status.replicas or 0}, ready: {d.status.ready_replicas or 0}"
    except ApiException as e: return _handle(e)

def describe_service(name):
    try:
        s = v1.read_namespaced_service(name=name.strip(), namespace=_ns())
        return f"🌐 Service '{s.metadata.name}' type: {s.spec.type}, cluster IP: {s.spec.cluster_ip}"
    except ApiException as e: return _handle(e)

def describe_job(name):
    try:
        j = batch_v1.read_namespaced_job(name=name.strip(), namespace=_ns())
        return f"⚙️ Job '{j.metadata.name}' completions: {j.status.succeeded or 0}, active: {j.status.active or 0}"
    except ApiException as e: return _handle(e)

def describe_configmap(name):
    try:
        cm = v1.read_namespaced_config_map(name=name.strip(), namespace=_ns())
        keys = list(cm.data.keys()) if cm.data else []
        return f"🗂️ ConfigMap '{cm.metadata.name}' keys: {keys}"
    except ApiException as e: return _handle(e)

def describe_secret(name):
    try:
        sec = v1.read_namespaced_secret(name=name.strip(), namespace=_ns())
        keys = list(sec.data.keys()) if sec.data else []
        return f"🔒 Secret '{sec.metadata.name}' contains {len(keys)} keys (values hidden)"
    except ApiException as e: return _handle(e)

def describe_pvc(name):
    try:
        pvc = v1.read_namespaced_persistent_volume_claim(name=name.strip(), namespace=_ns())
        cap = pvc.status.capacity.get('storage') if pvc.status and pvc.status.capacity else 'N/A'
        return f"💾 PVC '{pvc.metadata.name}' status: {pvc.status.phase}, capacity: {cap}"
    except ApiException as e: return _handle(e)

def describe_replicaset(name):
    try:
        rs = apps_v1.read_namespaced_replica_set(name=name.strip(), namespace=_ns())
        return f"📎 ReplicaSet '{rs.metadata.name}' replicas: {rs.status.replicas}, ready: {rs.status.ready_replicas}"
    except ApiException as e: return _handle(e)

def describe_statefulset(name):
    try:
        ss = apps_v1.read_namespaced_stateful_set(name=name.strip(), namespace=_ns())
        return f"📘 StatefulSet '{ss.metadata.name}' replicas: {ss.status.replicas}, ready: {ss.status.ready_replicas}"
    except ApiException as e: return _handle(e)

def describe_daemonset(name):
    try:
        ds = apps_v1.read_namespaced_daemon_set(name=name.strip(), namespace=_ns())
        return f"🔁 DaemonSet '{ds.metadata.name}' scheduled: {ds.status.current_number_scheduled}, ready: {ds.status.number_ready}"
    except ApiException as e: return _handle(e)

def describe_ingress(name):
    try:
        ing = networking_v1.read_namespaced_ingress(name=name.strip(), namespace=_ns())
        hosts = [r.host for r in (ing.spec.rules or []) if r.host]
        services = []
        for rule in ing.spec.rules or []:
            if rule.http:
                for path in rule.http.paths:
                    if path.backend and path.backend.service:
                        services.append(path.backend.service.name)
        return f"🚪 Ingress '{ing.metadata.name}' hosts: {hosts or []} → services: {services or []}"
    except ApiException as e: return _handle(e)

def describe_node(name):
    try:
        n = v1.read_node(name=name.strip())
        labels = n.metadata.labels or {}
        capacity = n.status.capacity or {}
        alloc = n.status.allocatable or {}
        gpu_cap = capacity.get('nvidia.com/gpu') or capacity.get('nvidia.com/mig-1g.5gb') or '0'
        gpu_alloc = alloc.get('nvidia.com/gpu') or alloc.get('nvidia.com/mig-1g.5gb') or '0'
        product = labels.get('nvidia.com/gpu.product') or labels.get('gpu.nvidia.com/product') or labels.get('accelerator', 'unknown')
        os_img = getattr(getattr(n.status, 'node_info', None), 'os_image', 'unknown')
        kernel = getattr(getattr(n.status, 'node_info', None), 'kernel_version', 'unknown')
        return (f"🖥️ Node '{n.metadata.name}'"
                f"  GPU product: {product}"
                f"  GPUs (capacity/allocatable): {gpu_cap}/{gpu_alloc}"
                f"  OS: {os_img}, Kernel: {kernel}")
    except ApiException as e: return _handle(e)

# ——— CONTEXT ———

def get_service_account(_=None):
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/serviceaccount.name", "r") as f:
            sa = f.read().strip()
            return f"Current service account: system:serviceaccount:{_ns()}:{sa}"
    except Exception:
        return f"Current service account in '{_ns()}' (name file not present)"


def check_permissions(_=None):
    try:
        from kubernetes.client import V1SelfSubjectAccessReview, V1SelfSubjectAccessReviewSpec, V1ResourceAttributes
        access_review = V1SelfSubjectAccessReview(
            spec=V1SelfSubjectAccessReviewSpec(
                resource_attributes=V1ResourceAttributes(namespace=_ns(), verb="list", resource="pods")
            )
        )
        resp = auth_v1.create_self_subject_access_review(access_review)
        return "✅ Can list pods" if resp.status.allowed else f"❌ Denied: {resp.status.reason}"
    except Exception as e:
        return f"❌ Error checking permissions: {e}"

# Registry to plug into agents
ACTIONS = {
    # ns
    "set_namespace": set_namespace,
    # list
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
    # describe
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
    # context
    "get_service_account": get_service_account,
    "check_permissions": check_permissions,
}