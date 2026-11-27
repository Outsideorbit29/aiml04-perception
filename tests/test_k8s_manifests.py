"""Static checks of the Minikube manifests in k8s/ (no cluster needed) and their consistency with
docker-compose.yml, configs/serving.yaml and the Feast Redis config."""

import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

K8S = Path("k8s")
NAMESPACE = "aiml01"
LOCAL_IMAGES = {"aiml01-services:dev", "aiml01-dashboard:dev"}


def _docs(path):
    return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]


@pytest.fixture(scope="module")
def objs():
    out = {}
    for p in sorted(K8S.glob("*.yaml")):
        if p.name == "kustomization.yaml":
            continue
        for d in _docs(p):
            out[(d["kind"], d["metadata"]["name"])] = d
    return out


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))["services"]


def _pod_spec(obj):
    return obj["spec"] if obj["kind"] == "Pod" else obj["spec"]["template"]["spec"]


def _workloads(objs):
    return [o for o in objs.values() if o["kind"] in ("Deployment", "Job", "Pod")]


def _main(objs, kind, name):
    return _pod_spec(objs[(kind, name)])["containers"][0]


def _mem_mi(q):
    m = re.fullmatch(r"(\d+)(Mi|Gi|m)?", str(q))
    assert m, q
    n, unit = int(m.group(1)), m.group(2)
    return n * 1024 if unit == "Gi" else n / 1024**2 if unit is None else n


def test_expected_objects(objs):
    for key in [
        ("Namespace", NAMESPACE),
        ("PersistentVolumeClaim", "aiml01-bundle"),
        ("ConfigMap", "aiml01-serving"),
        ("Deployment", "kafka"),
        ("Service", "kafka"),
        ("Deployment", "redis"),
        ("Service", "redis"),
        ("Deployment", "scorer"),
        ("Service", "scorer"),
        ("Deployment", "consumer"),
        ("Deployment", "dashboard"),
        ("Service", "dashboard"),
        ("Job", "producer"),
        ("Pod", "bundle-loader"),
    ]:
        assert key in objs, key


def test_kustomization_lists_the_stack():
    k = yaml.safe_load((K8S / "kustomization.yaml").read_text())
    files = {p.name for p in K8S.glob("*.yaml")} - {"kustomization.yaml"}
    assert set(k["resources"]) == files - {"producer-job.yaml", "bundle-loader.yaml"}
    assert all((K8S / r).exists() for r in k["resources"])


def test_namespaced(objs):
    for (kind, name), o in objs.items():
        if kind != "Namespace":
            assert o["metadata"].get("namespace") == NAMESPACE, (kind, name)


def test_selectors_and_service_ports(objs):
    deployments = {n: o for (k, n), o in objs.items() if k == "Deployment"}
    for name, d in deployments.items():
        assert d["spec"]["selector"]["matchLabels"].items() <= (
            d["spec"]["template"]["metadata"]["labels"].items()
        ), name
    for (kind, name), svc in objs.items():
        if kind != "Service":
            continue
        targets = [
            d
            for d in deployments.values()
            if svc["spec"]["selector"].items()
            <= d["spec"]["template"]["metadata"]["labels"].items()
        ]
        assert len(targets) == 1, name
        ports = {
            p["containerPort"] for c in _pod_spec(targets[0])["containers"] for p in c["ports"]
        }
        for p in svc["spec"]["ports"]:
            assert p["targetPort"] in ports, (name, p)
        if svc["spec"].get("type") == "NodePort":
            assert all(30000 <= p["nodePort"] <= 32767 for p in svc["spec"]["ports"])


def test_containers_images_resources(objs):
    for o in _workloads(objs):
        spec = _pod_spec(o)
        assert spec.get("enableServiceLinks") is False or o["kind"] == "Pod", o["metadata"]["name"]
        for c in spec["containers"] + spec.get("initContainers", []):
            name = f"{o['metadata']['name']}/{c['name']}"
            assert ":" in c["image"] and not c["image"].endswith(":latest"), name
            if c["image"] in LOCAL_IMAGES:
                assert c.get("imagePullPolicy") == "Never", name
            assert "memory" in c["resources"]["limits"], name
            req = c["resources"].get("requests", {}).get("memory")
            if req:
                assert _mem_mi(req) <= _mem_mi(c["resources"]["limits"]["memory"]), name


def test_fits_a_small_minikube(objs):
    """The long-running stack plus the producer job must fit a ~5 GB Minikube node."""
    req = lim = 0.0
    for o in _workloads(objs):
        if o["kind"] == "Pod":
            continue
        for c in _pod_spec(o)["containers"]:
            req += _mem_mi(c["resources"]["requests"]["memory"])
            lim += _mem_mi(c["resources"]["limits"]["memory"])
    assert req <= 3 * 1024 and lim <= 5 * 1024, (req, lim)


def test_volumes_reference_the_pvc(objs):
    for o in _workloads(objs):
        spec = _pod_spec(o)
        vols = {v["name"]: v for v in spec.get("volumes", [])}
        for c in spec["containers"]:
            for m in c.get("volumeMounts", []):
                assert vols[m["name"]]["persistentVolumeClaim"]["claimName"] == "aiml01-bundle"


def test_app_mounts_match_compose(objs, compose):
    """scorer and consumer get /app/models (read-only) and /app/data, like the compose x-app."""
    compose_mounts = {v.split(":")[1]: v for v in compose["scorer"]["volumes"]}
    for name in ("scorer", "consumer"):
        mounts = {m["mountPath"]: m for m in _main(objs, "Deployment", name)["volumeMounts"]}
        assert set(mounts) == set(compose_mounts), name
        assert mounts["/app/models"].get("readOnly") is True
        assert mounts["/app/models"]["subPath"] == "models"
        assert mounts["/app/data"]["subPath"] == "data"
    dash = {m["mountPath"]: m for m in _main(objs, "Deployment", "dashboard")["volumeMounts"]}
    assert dash["/app/data"]["subPath"] == "data" and not dash["/app/data"].get("readOnly")


def test_commands_match_compose(objs, compose):
    # k8s "args" replaces the image CMD (compose "command"), keeping the Feast entrypoint.
    for name in ("scorer", "consumer"):
        c = _main(objs, "Deployment", name)
        assert c["image"] == compose[name]["image"] == "aiml01-services:dev"
        assert c["args"] == compose[name]["command"] and "command" not in c
    # k8s "command" replaces the ENTRYPOINT, like compose's entrypoint override for the producer.
    p = _main(objs, "Job", "producer")
    assert p["command"] == compose["producer"]["entrypoint"]
    assert {"name": "FEAST_APPLY", "value": "0"} in p["env"]
    assert _main(objs, "Deployment", "redis")["args"] == compose["redis"]["command"]
    assert _main(objs, "Deployment", "redis")["image"] == compose["redis"]["image"]


def test_kafka_env_matches_compose(objs, compose):
    c = _main(objs, "Deployment", "kafka")
    assert c["image"] == compose["kafka"]["image"]
    env = {e["name"]: e["value"] for e in c["env"]}
    expected = {k: str(v) for k, v in compose["kafka"]["environment"].items()}
    assert set(env) == set(expected)
    for k, v in expected.items():
        if k == "KAFKA_CONTROLLER_QUORUM_VOTERS":
            assert env[k] == "1@localhost:9093"  # the controller is in the same pod
        else:
            assert env[k] == v, k
    assert _pod_spec(objs[("Deployment", "kafka")])["enableServiceLinks"] is False


def test_memory_limits_not_above_compose(objs, compose):
    for name in ("kafka", "redis", "scorer", "consumer"):
        lim = _main(objs, "Deployment", name)["resources"]["limits"]["memory"]
        assert _mem_mi(lim) <= _mem_mi(compose[name]["mem_limit"].replace("m", "Mi")), name


def test_service_dns_matches_serving_config(objs):
    serving = yaml.safe_load(Path("configs/serving.yaml").read_text())
    cm = objs[("ConfigMap", "aiml01-serving")]["data"]

    def port_of(svc):
        return {p["port"] for p in objs[("Service", svc)]["spec"]["ports"]}

    for bootstrap in (serving["kafka_bootstrap"], cm["KAFKA_BOOTSTRAP"]):
        host, port = bootstrap.split(":")
        assert int(port) in port_of(host)
    for url in (serving["scorer_url"], cm["SCORER_URL"]):
        u = urlparse(url)
        assert u.port in port_of(u.hostname)
    feast = yaml.safe_load(Path("feature_repo/feature_store.redis.yaml").read_text())
    host, port = feast["online_store"]["connection_string"].split(":")
    assert int(port) in port_of(host)


def test_configmap_keys_are_known_config_keys(objs):
    keys = set(yaml.safe_load(Path("configs/serving.yaml").read_text()))
    keys |= set(yaml.safe_load(Path("services/dashboard/dashboard.yaml").read_text()))
    cm = objs[("ConfigMap", "aiml01-serving")]["data"]
    assert {k.lower() for k in cm} <= keys
    for k, v in cm.items():
        if k.endswith(("_DB", "_PATH")):
            assert v.startswith("/app/data/"), k


def test_dashboard_port_matches_dockerfile(objs):
    dockerfile = Path("services/dashboard/Dockerfile").read_text()
    assert "--server.port=8501" in dockerfile and "EXPOSE 8501" in dockerfile
    c = _main(objs, "Deployment", "dashboard")
    assert c["image"] == "aiml01-dashboard:dev"
    assert c["ports"][0]["containerPort"] == 8501
    assert c["readinessProbe"]["httpGet"]["port"] == 8501
