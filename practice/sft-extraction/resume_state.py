"""Safe-container checkpoint primitives for the local controlled resume test."""
import hashlib
import json
import random
import numpy as np
import torch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def tree_digest(value):
    h = hashlib.sha256()

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            h.update(str(obj.dtype).encode())
            h.update(json.dumps(list(obj.shape)).encode())
            h.update(obj.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, dict):
            h.update(b"dict")
            for key in sorted(obj, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            h.update(type(obj).__name__.encode())
            for item in obj:
                visit(item)
        else:
            h.update(type(obj).__name__.encode())
            h.update(json.dumps(obj, ensure_ascii=False, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def capture_rng():
    kind, keys, position, gaussian, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": (kind, keys.tolist(), position, gaussian, cached),
            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    kind, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])
