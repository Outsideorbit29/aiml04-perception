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
2. **Firewall.** Allow inbound TCP 2000–2002 for `CarlaUE4-Win64-Shipping.exe` on the WSL virtual network.
3. **WSL2 + ROS2 Humble (ros-base, no GUI tools).**
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
   Inside Ubuntu:
   ```bash
   # follow docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html for the apt source, then:
   sudo apt update && sudo apt install -y ros-humble-ros-base python3-colcon-common-extensions ros-humble-cv-bridge
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   pip3 install carla==0.9.15 onnxruntime-gpu tensorrt opencv-python-headless numpy
   export CARLA_HOST=$(ip route | awk '/default/ {print $3}')
   python3 sim/smoke_test.py --host $CARLA_HOST
   ```
   Live inference uses ONNX Runtime or TensorRT only, with no PyTorch on the laptop.
4. **Limit WSL resources.** Create `C:\Users\<you>\.wslconfig`:
   ```ini
   [wsl2]
   memory=6GB
   processors=4
   ```

## 5. Keeping the laptop cool during CARLA sessions
- Plug in, keep the vents clear, and use the vendor app's balanced or quiet mode.
- Keep sessions to 30 minutes or less, with Town01/Town02, `-quality-level=Low`, `-RenderOffScreen`, and an 800×450 or smaller camera.
- Close the browser and other GPU apps. Stop CARLA as soon as a session ends.
- Watch temperatures with `nvidia-smi -l 5`. If the GPU stays above about 87 °C, stop and let it cool.

## 6. Known pitfalls
- WSL2 to Windows networking: if the CARLA client times out, check the firewall and use the host IP, not `localhost`.
- The CARLA 0.9.15 Python API supports Python 3.7–3.10 only (Ubuntu 22.04 ships 3.10).
- TensorRT engines built on Kaggle's T4 will **not** run on the RTX 3060. Rebuild locally from the ONNX files (M6.4).
- Kaggle sessions stop after idle time or the time limit. Always resume from the latest W&B checkpoint artifact.
