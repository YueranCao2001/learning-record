"""Read-only environment probe. Never reads or prints authentication values."""
import importlib.util
import json
import platform
import subprocess
import sys
from importlib.metadata import version


def inspect():
    packages = {}
    for name in ("torch", "transformers", "peft", "trl", "datasets", "huggingface_hub", "accelerate"):
        packages[name] = version(name) if importlib.util.find_spec(name) else None
    report = {"python": platform.python_version(), "executable": sys.executable,
              "packages": packages, "cuda_available": None, "torch_cuda": None,
              "auth_checked": False, "model_run": False}
    if packages["torch"]:
        import torch
        report.update(cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True,
                                timeout=15, check=False)
        report["nvidia_smi"] = {"returncode": result.returncode, "stdout": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi"] = {"error": type(exc).__name__}
    return report


if __name__ == "__main__":
    print(json.dumps(inspect(), ensure_ascii=False, indent=2))
