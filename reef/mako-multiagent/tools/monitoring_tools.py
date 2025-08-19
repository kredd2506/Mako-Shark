# tools/monitoring_tools.py
import os
import time
import json
import requests
from tabulate import tabulate

PROM_URL = os.environ.get("PROM_URL", "https://prometheus.nrp-nautilus.io/api/v1/query")

# ——— Concepts table (from a2) ———
CONCEPTS_TABLE = """
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

def show_kubernetes_concepts(_=None):
    return CONCEPTS_TABLE

# ——— PromQL helpers ———

def _query(promql: str):
    r = requests.get(PROM_URL, params={"query": promql}, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        return []
    return data["data"].get("result", [])


def namespace_gpu_utilization(raw_threshold: str = "0"):
    try:
        threshold = float(raw_threshold.strip() or 0)
    except Exception:
        threshold = 0
    promql = 'avg by (namespace) (DCGM_FI_DEV_GPU_UTIL)'
    rows = []
    for r in _query(promql):
        ns = r["metric"].get("namespace", "unknown")
        util = float(r["value"][1])
        if util >= threshold:
            status = "🟢 Low" if util < 40 else ("🟡 Moderate" if util < 70 else "🔴 High")
            rows.append([ns, f"{util:.2f}%", status])
    if not rows:
        return "✅ No namespaces above threshold"
    return tabulate(rows, headers=["Namespace", "Avg GPU Util", "Status"], tablefmt="fancy_grid")


def _fetch_dcgm_gpu_util():
    res = _query('DCGM_FI_DEV_GPU_UTIL')
    enriched = []
    for r in res:
        m = r.get("metric", {})
        val = float(r["value"][1])
        enriched.append({
            "hostname": m.get("Hostname", "unknown"),
            "gpu_id": m.get("gpu", "?"),
            "model": m.get("modelName", "unknown"),
            "namespace": m.get("namespace", "?"),
            "pod": m.get("pod", "?"),
            "util": val,
        })
    return enriched


def get_gpu_utilization_details(raw: str):
    parts = [p.strip() for p in (raw or "").split(',') if p.strip()]
    top_n = int(parts[0]) if parts else 10
    threshold = float(parts[1]) if len(parts) > 1 else 0
    data = [d for d in _fetch_dcgm_gpu_util() if d["util"] >= threshold]
    if not data:
        return f"✅ No GPUs over {threshold}% utilization."
    data.sort(key=lambda x: x["util"], reverse=True)
    data = data[:top_n]
    rows = [[d['hostname'][:20], d['gpu_id'], d['model'][:25], f"{d['util']:.2f}%", d['namespace'], d['pod'][:24]] for d in data]
    return tabulate(rows, headers=["Host","GPU","Model","Util%","NS","Pod"], tablefmt="grid")


def get_gpu_utilization_stats(raw_threshold: str = "0"):
    try:
        threshold = float(raw_threshold.strip() or 0)
    except Exception:
        threshold = 0
    data = [d for d in _fetch_dcgm_gpu_util() if d["util"] >= threshold]
    if not data:
        return f"✅ No GPUs over {threshold}% utilization."
    n = len(data)
    avg = sum(d['util'] for d in data) / n
    maxed = len([1 for d in data if d['util'] >= 99])
    idle = len([1 for d in data if d['util'] < 1])
    moderate = len([1 for d in data if 1 <= d['util'] < 70])
    hosts = len(set(d['hostname'] for d in data))
    models = len(set(d['model'] for d in data))
    avail = len([1 for d in data if d['util'] < 100])
    return (f"📊 GPU Utilization Stats (threshold: {threshold}%):\n"
            f"🔍 Total GPUs: {n}\n"
            f"📈 Average Utilization: {avg:.2f}%\n"
            f"🔴 Fully Utilized (>=99%): {maxed}\n"
            f"🟢 Idle (<1%): {idle}\n"
            f"⚙️ Moderate (1-70%): {moderate}\n"
            f"💻 Unique Hosts: {hosts}\n"
            f"🧠 Unique Models: {models}\n"
            f"🧮 Available (<100%): {avail}")


#added new action for Prometheus ping
def prom_ping(_=None):
    try:
        r = requests.get(os.environ.get("PROM_URL","https://prometheus.nrp-nautilus.io/api/v1/query"),
                         params={"query":"up"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        n = len(data.get("data",{}).get("result",[])) if data.get("status")=="success" else 0
        return f"✅ up targets: {n}"
    except Exception as e:
        return f"❌ Prometheus error: {e}"


def prom_raw(promql: str):
     promql = (promql or "").strip() or "up"
     try:
         res = _query(promql)
        # show a compact preview
         preview = [
             {k: v for k, v in r.get("metric", {}).items() if k in ("job","namespace","pod","instance")}
             for r in res[:5]
         ]
         return f"OK {len(res)} results\n" + json.dumps(preview, indent=2)
     except Exception as e:
         return f"❌ {e}"
     
def get_gpu_model_summary(raw: str = "A100"):
    model = (raw or "A100").strip()
    data = _fetch_dcgm_gpu_util()
    subset = [d for d in data if model.lower() in d['model'].lower()]
    if not subset:
        return f"✅ No GPUs matching '{model}'."
    total = len(subset)
    hosts = len(set(d['hostname'] for d in subset))
    idle = sum(1 for d in subset if d['util'] < 1)
    busy = sum(1 for d in subset if d['util'] >= 90)
    avg = sum(d['util'] for d in subset) / total
    top = sorted(subset, key=lambda x: x['util'], reverse=True)[:10]
    rows = [[d['hostname'][:22], d['gpu_id'], f"{d['util']:.1f}%", d['namespace'], d['pod'][:28]] for d in top]
    table = tabulate(rows, headers=["Host","GPU","Util%","NS","Pod"], tablefmt="fancy_grid")
    return (f"🧬 Model '{model}' summary\n"
            f"Total: {total} GPUs across {hosts} hosts\n"
            f"Avg util: {avg:.1f}% | Busy (>=90%): {busy} | Idle (<1%): {idle}" + table)


def list_gpu_nodes_for_model(raw: str = "A100"):
    model = (raw or "A100").strip()
    data = _fetch_dcgm_gpu_util()
    subset = [d for d in data if model.lower() in d['model'].lower()]
    if not subset:
        return f"✅ No GPUs matching '{model}'."
    counts = {}
    for d in subset:
        counts[d['hostname']] = counts.get(d['hostname'], 0) + 1
    rows = sorted(((h, c) for h, c in counts.items()), key=lambda x: (-x[1], x[0]))
    return tabulate([[h, c] for h, c in rows], headers=["Host","# of GPUs"], tablefmt="github")


# Registry for agents
ACTIONS = {
    "show_kubernetes_concepts": show_kubernetes_concepts,
    "namespace_gpu_utilization": namespace_gpu_utilization,
    "get_gpu_utilization_details": get_gpu_utilization_details,
    "get_gpu_utilization_stats": get_gpu_utilization_stats,
    
    #added newer action for Prometheus ping
    "prom_ping": prom_ping,
    "prom_raw": prom_raw,
    #new
    "get_gpu_model_summary": get_gpu_model_summary,
    "list_gpu_nodes_for_model": list_gpu_nodes_for_model,
}