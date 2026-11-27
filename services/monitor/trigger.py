"""Retraining trigger: launch the F3 Kaggle training job when a drift decision says so.

Dry-run by default: the `kaggle kernels push -p kaggle/train` command is printed and recorded,
not run. With execute=True it is run, at most once per `cooldown_hours` (state in a JSON file).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def retrain_command(trigger_cfg: Mapping[str, Any]) -> list[str]:
    cli = os.environ.get("KAGGLE") or trigger_cfg["kaggle_cli"]
    return [cli, "kernels", "push", "-p", trigger_cfg["kernel_dir"]]


def _load_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def maybe_retrain(
    decision: Mapping[str, Any],
    trigger_cfg: Mapping[str, Any],
    cooldown_hours: float,
    execute: bool = False,
    now: float | None = None,
    runner: Callable[[list[str]], int] | None = None,
) -> dict[str, Any]:
    """Act on a `decide_retrain` result. Returns what happened (never raises on launch errors)."""
    now = time.time() if now is None else now
    cmd = retrain_command(trigger_cfg)
    out: dict[str, Any] = {"action": "none", "command": shlex.join(cmd), "at": now}
    if not decision.get("retrain"):
        return out
    state_path = Path(trigger_cfg["state_path"])
    state = _load_state(state_path)
    last = state.get("last_launch")
    if last is not None and now - float(last) < cooldown_hours * 3600:
        out["action"] = "cooldown"
        out["next_allowed"] = float(last) + cooldown_hours * 3600
        return out
    if not execute:
        out["action"] = "dry_run"
        print(
            f"[dry-run] would retrain: {out['command']}  ({'; '.join(decision['reasons'])})",
            flush=True,
        )
        return out
    run = runner or (lambda c: subprocess.run(c, check=False).returncode)
    try:
        code = run(cmd)
    except OSError as exc:  # CLI missing (e.g. inside the serving container)
        out.update(action="failed", error=f"{type(exc).__name__}: {exc}")
        return out
    out.update(action="launched" if code == 0 else "failed", returncode=code)
    if code == 0:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"last_launch": now, "reasons": list(decision.get("reasons", []))}, indent=2)
        )
    return out
