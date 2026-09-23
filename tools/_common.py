"""Shared paths and helpers for the reference tooling.

The tools import the training package (`experiment`) directly so the mock
checkpoint and the fixtures are produced by exactly the code that will
produce the real published model.
"""

import hashlib
import os
import sys
from pathlib import Path

CLASSIFICATION_MODELS = Path(
    os.environ.get("VAKT_CLASSIFICATION_MODELS", "/Users/shearer/vaktex/classification_models")
)
BASE_MODEL = CLASSIFICATION_MODELS / "Qwen3.5-0.8B-Base"
TESTDATA = Path(
    os.environ.get("VAKT_TESTDATA", str(CLASSIFICATION_MODELS / "Harness" / "testdata"))
)
MOCK_FP32 = TESTDATA / "models" / "mock-dom-0.8b" / "model.safetensors"
MOCK_BF16 = TESTDATA / "models" / "mock-dom-0.8b-bf16" / "model.safetensors"
PARITY_DIR = TESTDATA / "parity"

# Number of auxiliary CWE-family outputs of the published model.
AUXILIARY_OUTPUTS = 18

if str(CLASSIFICATION_MODELS) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_MODELS))


def sha256_file(path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def build_classifier(base: Path = BASE_MODEL):
    """Build the classifier exactly as training does, on a CPU float32 backend.

    Gradient checkpointing stays enabled as in training; it is inactive in
    eval mode and never changes a weight.
    """
    from experiment.config import ExperimentConfig, detect_backend
    from experiment.model import QwenClassifier

    config = ExperimentConfig(model_path=str(base))
    backend = detect_backend(prefer="cpu")
    model = QwenClassifier(config, backend, auxiliary_outputs=AUXILIARY_OUTPUTS)

    return model, config, backend
