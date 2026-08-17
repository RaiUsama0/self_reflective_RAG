"""Central configuration: paths, defaults, and per-run settings.

Every path derives from PROJECT_ROOT, so nothing depends on the directory a script
happens to be launched from. Every run is described by one RunConfig, written next
to that run's results so it stays reproducible.
"""
from __future__ import annotations

import json
import os
import time
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

DEFAULT_GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "openai:gpt-4o-mini")
DEFAULT_VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL", "")


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
    expand_k_each_iteration: int = 3
    max_iterations: int = 3


@dataclass
class GenerationConfig:
    model: str = DEFAULT_LLM
    max_new_tokens: int = 320
    temperature: float = 0.0
    load_in_4bit: bool = False


@dataclass
class ModelConfig:
    """Records exactly which model played which role - the answer to "which model
    was used for generation / decomposition / verification / reformulation" that
    ISSUE 1 requires every run to make explicit. Decomposition and verification are
    both the verifier's responsibility, so they share `verifier`; reformulation is a
    generative task (drafting a search query, not judging evidence), so it shares
    `generator`."""

    generator: str = DEFAULT_GENERATOR_MODEL
    verifier: str = ""
    reformulator: str = ""
    verifier_is_independent: bool = False

    def __post_init__(self) -> None:
        if not self.verifier:
            self.verifier = self.generator
        if not self.reformulator:
            self.reformulator = self.generator
        self.verifier_is_independent = self.verifier != self.generator


@dataclass
class RunConfig:
    name: str = "baseline"
    arm: str = "baseline"
    subset: str = ""
    subset_checksum: str = ""
    index_dir: str = ""
    seed: int = DEFAULT_SEED
    dataset: str = ""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    min_support_ratio: float = 1.0
    timestamp_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

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
            models=ModelConfig(**d.pop("models", {})),
            **d,
        )
