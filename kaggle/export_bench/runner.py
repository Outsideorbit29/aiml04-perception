"""Kaggle runner for M6.1-M6.3: ONNX export + parity, T4 TensorRT FP16 engines, benchmark (GPU).

Set MODE (and REF if the code is on a branch), push the code, then from the laptop:
    kaggle kernels push -p kaggle/export_bench
    smoke -> 10 + 50 timed iterations, 40 val images: proves export, engine build and every
             benchmark row (~20-25 min incl. pip install); check export.json + benchmarks.md
    full  -> 200 warm-up + 1000 timed iterations per row, 500 val images (~45-60 min)
Steps (configs/bench.yaml): deploy/export.py (YOLO11s from the W&B artifact, MiDaS-small from
torch.hub, YOLOP ONNX from hustvl/YOLOP; ONNX files logged as W&B artifacts), then
deploy/build_local_engine.py on the T4 (same script as the laptop build in M6.4), then
deploy/benchmark.py. BDD val (accuracy delta) comes from the data-prep kernel output. Small
outputs land in /kaggle/working/m6_bench (export.json, engines.json, results.json,
benchmarks.md); the ONNX files and engines stay in /kaggle/temp (W&B holds the ONNX files).
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "m6-onnx-trt"  # branch, tag or commit SHA to run; set back to "main" before merging
MODE = "full"  # smoke | full
EXTRA = ["--smoke"] if MODE == "smoke" else []
MODELS_DIR = "/kaggle/temp/m6_models"  # configs/bench.yaml models_dir
OUT_DIR = "/kaggle/working/m6_bench"  # configs/bench.yaml out_dir
JOB_CMDS = [
    ["python", "deploy/export.py", "--config", "configs/bench.yaml", *EXTRA],
    ["python", "deploy/build_local_engine.py", "--onnx", f"{MODELS_DIR}/*.onnx", "--fp16",
     "--workspace-mib", "1024", "--summary", f"{OUT_DIR}/engines.json"],
    ["python", "deploy/benchmark.py", "--config", "configs/bench.yaml", *EXTRA],
]
# timm 0.6.13: intel-isl/MiDaS hubconf (as in kaggle/eval_depth). tensorrt-cu12 + cuda-python:
# engine build and the torch-free TensorRT runtime. nvidia-ml-py: VRAM. yacs/prefetch_generator:
# hustvl/YOLOP hubconf imports. onnxslim: Ultralytics ONNX simplify.
JOB_PIP = [
    "timm==0.6.13", "tensorrt-cu12>=10.0,<11", "cuda-python>=12.0,<13", "nvidia-ml-py",
    "onnxslim", "yacs", "prefetch_generator",
]
WORKDIR = "/kaggle/temp/repo"  # outside /kaggle/working so the code is not saved as job output


def sh(cmd: list[str], check: bool = True, **kw) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, **kw).returncode


def load_wandb_key() -> None:
    """Find the W&B key, trying three sources in order.

    1. Kaggle Secrets (works for runs saved from the browser editor).
    2. A file named `wandb_api_key` in an attached private Kaggle dataset
       (works for runs pushed with `kaggle kernels push`, which get no secrets).
    3. Neither: log offline into /kaggle/working/wandb so the run is saved as job output
       and can be uploaded later with `wandb sync`.
    """
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
        print("W&B key loaded from Kaggle Secrets")
        return
    except Exception as exc:
        print(f"Kaggle Secrets unavailable ({type(exc).__name__})")
    # Shallow patterns only: a recursive glob would walk the 60k-file data-prep output.
    patterns = ["/kaggle/input" + "/*" * depth + "/wandb_api_key" for depth in range(1, 5)]
    for path in [p for pat in patterns for p in sorted(glob.glob(pat))]:
        with open(path) as fh:
            key = fh.read().strip()
        if key:
            os.environ["WANDB_API_KEY"] = key
            print("W&B key loaded from attached private dataset")
            return
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "/kaggle/working"
    print("No W&B key found; logging offline to /kaggle/working/wandb (sync later)")


def main() -> None:
    load_wandb_key()
    if not os.path.exists(WORKDIR):
        sh(["git", "clone", "--depth", "50", REPO, WORKDIR])
    # A shallow clone has only the default branch, so fetch REF explicitly.
    sh(["git", "fetch", "--depth", "50", "origin", REF], cwd=WORKDIR)
    sh(["git", "checkout", "FETCH_HEAD"], cwd=WORKDIR)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKDIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    os.environ["GIT_COMMIT"] = sha
    print(f"code commit {sha} (REF {REF}, MODE {MODE})", flush=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=WORKDIR)
    sh([sys.executable, "-m", "pip", "install", "-q", *JOB_PIP], cwd=WORKDIR)
    sh(["nvidia-smi"])
    os.makedirs(OUT_DIR, exist_ok=True)
    # One GPU for every step: NVML VRAM readings are taken on device 0.
    env = {**os.environ, "PYTHONPATH": WORKDIR, "CUDA_VISIBLE_DEVICES": "0"}
    # Run every step even if one fails (the benchmark reports missing files as failed rows),
    # then fail the job so the status shows it.
    failed = [cmd[1] for cmd in JOB_CMDS if sh(cmd, check=False, cwd=WORKDIR, env=env) != 0]
    if failed:
        raise SystemExit(f"failed steps: {failed}")


if __name__ == "__main__":
    main()
