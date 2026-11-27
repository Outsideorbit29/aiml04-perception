# Deployment (F7): Docker Compose and Minikube

Two ways to run the serving stack. Both use the serving bundle from
`scripts/fetch_serving_bundle.sh` in `./serving_bundle/` (`models/` + `data/`, never committed).

| Component | Compose service | Kubernetes object (namespace `aiml01`) | Image |
|---|---|---|---|
| Kafka (KRaft, single node) | `kafka` | Deployment + Service `kafka:9092` | `apache/kafka:3.8.0` |
| Feast online store | `redis` | Deployment + Service `redis:6379` | `redis:7.2-alpine` |
| FastAPI scorer | `scorer` | Deployment + Service `scorer:8000` | `aiml01-services:dev` |
| Stream consumer | `consumer` | Deployment (1 replica, the case DB writer) | `aiml01-services:dev` |
| Replay producer | `producer` (profile `replay`) | Job `producer` | `aiml01-services:dev` |
| Analyst dashboard | `dashboard` | Deployment + NodePort Service `30851` | `aiml01-dashboard:dev` |
| Bundle + case DB | `./serving_bundle` bind mounts | PVC `aiml01-bundle` (`models/`, `data/` sub-paths) | – |

Service names equal the compose names, so `configs/serving.yaml` and
`feature_repo/feature_store.redis.yaml` work unchanged; the ConfigMap `aiml01-serving` repeats
the overrides (`KAFKA_BOOTSTRAP`, `SCORER_URL`, `CASES_DB`, ...) so they can be changed without
rebuilding. `tests/test_k8s_manifests.py` checks the manifests against compose and the configs.

## Kubernetes manifests (`k8s/`)
| File | Contents |
|---|---|
| `namespace.yaml` | namespace `aiml01` |
| `storage.yaml` | PVC `aiml01-bundle` (1 Gi, Minikube `standard` hostPath class) |
| `configmap.yaml` | env overrides for `configs/serving.yaml` |
| `kafka.yaml`, `redis.yaml` | as in compose; Kafka data is ephemeral, like compose |
| `scorer.yaml`, `consumer.yaml` | `args` replaces the image CMD only, so `services/entrypoint.sh` still runs `feast apply`; init containers wait for Kafka/Redis/scorer (compose `depends_on`) |
| `dashboard.yaml` | Streamlit dashboard, NodePort 30851 |
| `producer-job.yaml` | one-off replay Job (`--rate 50 --limit 5000`), not in the kustomization |
| `bundle-loader.yaml` | temporary busybox pod for copying the bundle into the PVC |
| `kustomization.yaml` | the long-running stack (`kubectl apply -k k8s/`) |

Memory: requests ~2.4 GiB, limits ~4.8 GiB including the producer Job, all below the compose
`mem_limit`s. All pods disable `enableServiceLinks`: otherwise Kubernetes injects
`KAFKA_PORT=tcp://...` into the Kafka pod, which the image would turn into a broker property.
In the pod the KRaft quorum voter is `1@localhost:9093` (compose: `1@kafka:9093`).

## Minikube run (Git Bash, from the AIML-01 checkout)
Prerequisites: Docker Desktop, `minikube` and `kubectl` on PATH, the bundle in `./serving_bundle`.
Stop the compose stack first (`docker compose down`), since both share the WSL memory cap
(`.wslconfig` is 6 GB, so Minikube gets 4.5 GB).

```bash
# 1. Cluster (docker driver, small)
minikube start --driver=docker --cpus=4 --memory=4600m --disk-size=15g

# 2. Images: build on the host, then load into Minikube (imagePullPolicy: Never)
docker build -f services/Dockerfile -t aiml01-services:dev .
docker build -f services/dashboard/Dockerfile -t aiml01-dashboard:dev .
minikube image load aiml01-services:dev
minikube image load aiml01-dashboard:dev

# 3. Namespace, PVC, and copy the serving bundle into it
kubectl apply -f k8s/namespace.yaml -f k8s/storage.yaml -f k8s/bundle-loader.yaml
kubectl -n aiml01 wait --for=condition=Ready pod/bundle-loader --timeout=120s
for f in serving_bundle/models/*; do
  kubectl cp "$f" "aiml01/bundle-loader:/bundle/models/$(basename "$f")"
done
for f in history replay rings; do       # not cases.db: the cluster starts its own case DB
  kubectl cp "serving_bundle/data/$f.parquet" "aiml01/bundle-loader:/bundle/data/$f.parquet"
done
kubectl -n aiml01 exec bundle-loader -- ls -la /bundle/models /bundle/data
kubectl -n aiml01 delete pod bundle-loader

# 4. Stack
kubectl apply -k k8s/
for d in kafka redis scorer consumer dashboard; do
  kubectl -n aiml01 rollout status deploy/$d --timeout=300s
done
kubectl -n aiml01 logs -f deploy/consumer        # wait for "consumer ready" (feature warm-up)

# 5. Replay transactions
kubectl apply -f k8s/producer-job.yaml
kubectl -n aiml01 logs -f job/producer

# 6. Open the dashboard and the scorer API
minikube service dashboard -n aiml01 --url       # NodePort 30851; keep this terminal open
kubectl -n aiml01 port-forward svc/scorer 8000:8000   # http://localhost:8000/docs
# (or: kubectl -n aiml01 port-forward svc/dashboard 8501:8501 -> http://localhost:8501)

# Status / debugging
kubectl -n aiml01 get pods,svc,pvc
kubectl -n aiml01 describe pod -l app=consumer
kubectl top pods -n aiml01                       # needs: minikube addons enable metrics-server

# Run the replay again
kubectl -n aiml01 delete job producer && kubectl apply -f k8s/producer-job.yaml

# 7. Teardown
kubectl delete namespace aiml01                  # removes everything incl. the PVC data
minikube stop                                    # or: minikube delete (frees the disk too)
```

Notes
- With the docker driver on Windows, a NodePort is not reachable at `$(minikube ip):30851`
  directly; `minikube service ... --url` opens a tunnel and prints a `127.0.0.1` URL.
- After changing code, rebuild the image, `minikube image load` it again and run
  `kubectl -n aiml01 rollout restart deploy/scorer deploy/consumer deploy/dashboard`.
- The case DB is SQLite on the PVC: keep one consumer replica. Moving to Postgres would be the
  step to scale consumers horizontally.
