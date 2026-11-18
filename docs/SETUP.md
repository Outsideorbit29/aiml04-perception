# SETUP.md – Development environment (cloud-first)

Heavy work runs online. The laptop (Windows 11, RTX 3060 Laptop 6 GB) only runs CARLA, ROS2 and
the final TensorRT engine. Expected laptop disk use is about 25 GB, mostly CARLA and Ubuntu.

## 1. Online accounts (do these yourself, no installs)
1. **GitHub.** Create a public repo `aiml04-perception` and push this folder. Kaggle notebooks clone it.
2. **Kaggle.** Sign up at kaggle.com and verify your phone number under Settings. GPU and internet access in notebooks need this.
   - Settings → API Tokens → **Generate New Token** (the recommended option, not "Legacy"). Save the token as the file `C:\Users\<you>\.kaggle\access_token` (one line, the token only), or set it as the user environment variable `KAGGLE_API_TOKEN`. Never commit it.
   - W&B key for jobs: Kaggle Secrets are not passed to jobs pushed from the CLI. Instead, save the key (from wandb.ai/authorize) as `%USERPROFILE%\.kaggle\aiml04-secrets\wandb_api_key` and upload that folder once as a **private** dataset with `kaggle datasets create -p %USERPROFILE%\.kaggle\aiml04-secrets`. Every job attaches `anish2906/aiml04-secrets`.
3. **Weights & Biases.** Sign up at wandb.ai and create the project `aiml04-perception`.
4. **Hugging Face** (optional, M7.8): account for a free CPU Space demo.

## 2. Laptop: light tools (Windows)
```powershell
pip install kaggle wandb          # small CLIs: launch Kaggle jobs, read results
# pip puts kaggle.exe in %APPDATA%\Python\Python314\Scripts; add that folder to your user PATH
kaggle kernels list --mine        # check the token works
```
A full PyTorch/CUDA install on Windows is **not** needed. Unit tests run in GitHub Actions. For
quick local checks, a CPU-only Python 3.10 venv is enough:
```powershell
py -3.10 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt   # numpy, opencv-python-headless, pytest, ruff (no torch)
```

## 3. How a cloud job runs
```
laptop: git push  ─►  kaggle kernels push -p kaggle/train_yolo  ─►  Kaggle GPU runs the notebook
                                                                       │ clones repo @ commit
                                                                       │ runs scripts/train_yolo.py
                                                                       ▼
laptop: check the W&B dashboard  ◄─  metrics + best.pt/ONNX artifacts  ◄─  W&B
```
Each folder under `kaggle/` holds `kernel-metadata.json` (GPU on, internet on, datasets attached)
and a short notebook that clones the repo and calls one script. Download outputs with
`kaggle kernels output <user>/<kernel> -p runs/` only when you need them locally (the ONNX files for M6.4).

Kaggle limits: about 30 GPU-hours per week and up to 12 h per session (check your account's
current quota). Training saves a checkpoint every epoch and resumes from the last W&B artifact.

## 4. Laptop: simulator (only for M5/M6.4)
1. **CARLA 0.9.15 base package only.** Download `CARLA_0.9.15.zip` from the CARLA GitHub releases (skip AdditionalMaps) and extract it to `D:\CARLA_0.9.15`.
   ```powershell
   D:\CARLA_0.9.15\CarlaUE4.exe -quality-level=Low -RenderOffScreen -fps=20
   ```
2. **WSL2 resources and networking.** Create `C:\Users\<you>\.wslconfig`, then run `wsl --shutdown` once:
   ```ini
   [wsl2]
   memory=6GB
   processors=4
   swap=4GB
   networkingMode=mirrored
   ```
   With mirrored networking WSL2 shares the Windows network stack, so the CARLA server is at
   `localhost:2000` from inside Ubuntu (the default `carla.host` in `configs/carla.yaml`). No
   firewall rule and no host-IP lookup are needed.
3. **WSL2 + ROS2 Humble (ros-base, no GUI tools) + CARLA client.** One script installs ROS2
   Humble `ros-base`, `cv_bridge`, Cyclone DDS, colcon, the `carla==0.9.15` client, OpenCV/numpy 1.x, and the
   live runtime (`onnxruntime-gpu`, `supervision` for ByteTrack; skip it with `WITH_RUNTIME=0`).
   No PyTorch. Safe to re-run.
   ```powershell
   # Windows (PowerShell)
   wsl --install -d Ubuntu-22.04
   wsl -d Ubuntu-22.04 -u root -- bash /mnt/d/Project_ai/sim/setup_wsl.sh
   ```
4. **Smoke test (M0.8).** Start CARLA on Windows (step 1), then in WSL2:
   ```bash
   # WSL2 Ubuntu 22.04
   cd /mnt/d/Project_ai
   python3 sim/smoke_test.py --town Town01 --out runs/smoke/frame.png   # prints SMOKE_OK
   ```
   It switches the world to synchronous mode, spawns the ego and an 800x450 camera, saves one
   frame, then destroys the actors and restores the settings (also on errors and Ctrl+C).
5. **Build the ROS2 packages (M5).** The build lives in `~/aiml04_ws` (Linux disk) with symlinks
   back to `ros2_ws/src`, so nothing is written into the repo and Python edits need no rebuild:
   ```bash
   # WSL2 Ubuntu 22.04
   bash /mnt/d/Project_ai/ros2_ws/build_wsl.sh
   source /opt/ros/humble/setup.bash && source ~/aiml04_ws/install/setup.bash
   export AIML04_REPO_ROOT=/mnt/d/Project_ai RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=1
   ros2 launch perception_ros stack.launch.py carla_host:=localhost town:=Town01 load_town:=true \
       scenario:=stopped_car          # stopped_car | pedestrian_crossing | cut_in | night_rain
   ```
   Live inference uses ONNX Runtime (TensorRT after M6.4) with the M6 ONNX files in `models/`
   (`yolo11s-bdd.onnx`, `midas_small.onnx`, `yolop-640-640.onnx`; paths in
   `configs/pipeline_carla.yaml`). Scenario results are appended to
   `runs/scenarios/results.jsonl`, videos go to `media/`.

## 5. Keeping the laptop cool during CARLA sessions
- Plug in, keep the vents clear, and use the vendor app's balanced or quiet mode.
- Keep sessions to 30 minutes or less, with Town01/Town02, `-quality-level=Low`, `-RenderOffScreen`, and an 800×450 or smaller camera.
- Close the browser and other GPU apps. Stop CARLA as soon as a session ends.
- Watch temperatures with `nvidia-smi -l 5`. If the GPU stays above about 87 °C, stop and let it cool.

## 6. Known pitfalls
- WSL2 to Windows networking: with `networkingMode=mirrored` use `localhost`. Without mirrored mode (NAT), use the host IP from `ip route | awk '/default/ {print $3}'` and allow inbound TCP 2000–2002 for `CarlaUE4-Win64-Shipping.exe` in the Windows firewall.
- ROS2 nodes that run but never receive messages: with mirrored networking `lo` is not multicast-capable and the default Fast DDS finds no peers. Use Cyclone DDS on loopback: `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=1` (`sim/setup_wsl.sh` installs it and adds both lines to `~/.bashrc`; `stack.launch.py` sets them for its nodes). Use the same settings in any shell where you run `ros2 topic echo`.
- Git on Windows converts line endings; `.gitattributes` keeps `*.sh`, `*.msg`, `sim/` and `ros2_ws/` at LF so the scripts run in bash.
- If a run is killed hard (WSL closed), CARLA can stay in synchronous mode and look frozen: restart `CarlaUE4.exe`. A normal shutdown (Ctrl+C) destroys the actors and restores asynchronous mode.
- The CARLA 0.9.15 Python API supports Python 3.7–3.10 only (Ubuntu 22.04 ships 3.10).
- TensorRT engines built on Kaggle's T4 will **not** run on the RTX 3060. Rebuild locally from the ONNX files (M6.4).
- Kaggle sessions stop after idle time or the time limit. Always resume from the latest W&B checkpoint artifact.
