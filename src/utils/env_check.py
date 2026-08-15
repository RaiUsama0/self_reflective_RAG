"""Toolchain verification.

Run this first on any new machine or Colab session and paste the output into your
lab notebook. "Which environment produced this result" is a viva question.
"""
from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = ["numpy", "sklearn"]
OPTIONAL = [
    ("torch", "PyTorch - model inference"),
    ("transformers", "HuggingFace - generator models"),
    ("sentence_transformers", "dense passage/query embeddings"),
    ("faiss", "vector index (numpy fallback otherwise)"),
    ("datasets", "HotpotQA / FEVER download"),
    ("accelerate", "device placement for large models"),
    ("langchain", "listed in the project plan; not used by the pipeline"),
    ("bitsandbytes", "4-bit loading when GPU memory is tight"),
    ("matplotlib", "plots"),
]


def _version(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return False, f"not installed ({type(exc).__name__})"
    return True, getattr(mod, "__version__", "unknown")


def check_environment() -> int:
    print(f"python   {sys.version.split()[0]}   {platform.platform()}\n")

    missing = []
    print("required:")
    for name in REQUIRED:
        ok, ver = _version(name)
        print(f"  {name:<24} {ver}")
        if not ok:
            missing.append(name)

    print("\noptional:")
    for name, why in OPTIONAL:
        _, ver = _version(name)
        print(f"  {name:<24} {ver:<28} {why}")

    print("\ngpu:")
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  cuda:{i}  {p.name}  {p.total_memory / 1e9:.1f} GB")
            print(f"  torch cuda build: {torch.version.cuda}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("  Apple MPS available (no CUDA)")
        else:
            print("  no GPU visible - use a smaller model, --llm openai:<model>, or Colab")
    except ImportError:
        print("  torch not installed")

    print("\nfaiss:")
    try:
        import faiss
        import numpy as np

        idx = faiss.IndexFlatIP(8)
        idx.add(np.random.rand(10, 8).astype("float32"))
        print(f"  ok - built a {idx.ntotal}-vector index; "
              f"{faiss.get_num_gpus()} GPU(s) visible to faiss")
    except ImportError:
        print("  not installed - exact numpy index will be used "
              "(fine below ~1e5 passages)")
    except Exception as exc:
        print(f"  installed but failed: {exc}")

    if missing:
        print(f"\nMISSING REQUIRED: {', '.join(missing)}")
        print(f"  pip install {' '.join(missing)}")
        return 1
    print("\ncore dependencies present.")
    return 0
