"""Central configuration: paths, defaults, and per-run settings.

Every path derives from PROJECT_ROOT, so nothing depends on the directory a script
happens to be launched from. Every run is described by one RunConfig, written next
to that run's results so it stays reproducible.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SUBSETS_DIR = DATA_DIR / "subsets"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (RAW_DIR, PROCESSED_DIR, SUBSETS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "mock"
DEFAULT_SEED = 13
DEFAULT_TOP_K = 5


@dataclass
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDER
    batch_size: int = 64
    device: str | None = None
    normalize: bool = True


@dataclass
class RetrievalConfig:
    top_k: int = DEFAULT_TOP_K
    index_type: str = "flat"
    nlist: int = 100
    nprobe: int = 10
    backend: str = "auto"


@dataclass
class GenerationConfig:
    model: str = DEFAULT_LLM
    max_new_tokens: int = 320
    temperature: float = 0.0
    load_in_4bit: bool = False


@dataclass
class RunConfig:
    name: str = "baseline"
    subset: str = ""
    index_dir: str = ""
    seed: int = DEFAULT_SEED
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        d = json.loads(Path(path).read_text())
        return cls(
            embedding=EmbeddingConfig(**d.pop("embedding", {})),
            retrieval=RetrievalConfig(**d.pop("retrieval", {})),
            generation=GenerationConfig(**d.pop("generation", {})),
            **d,
        )
