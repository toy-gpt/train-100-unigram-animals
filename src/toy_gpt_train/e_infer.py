"""e_infer.py - Inference module (artifact-driven).

Runs inference using previously saved training artifacts.

Responsibilities:
- Load inspectable training artifacts from artifacts/
  - 00_meta.json
  - 01_vocabulary.csv
  - 02_model_weights.csv
- Reconstruct a vocabulary-like interface and model weights
- Generate tokens using greedy decoding (argmax)
- Print top-k next-token probabilities for inspection

Notes:
- This module does NOT retrain by default.
- If artifacts are missing, run d_train.py first.

Unigram inference:
    The model ignores all context and predicts based solely on
    corpus word frequencies. Every call to forward() returns the
    same probability distribution.
"""

import argparse
import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

from toy_gpt_train.c_model import SimpleNextTokenModel
from toy_gpt_train.e_infer import (
    ArtifactVocabulary,
    generate_tokens_unigram,
    load_meta,
    load_model_weights_csv,
    load_vocabulary_csv,
    require_artifacts,
    top_k,
)
from toy_gpt_train.prompts import parse_args

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

LOG: logging.Logger = get_logger("INFER", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR: Final[Path] = BASE_DIR / "artifacts"
META_PATH: Final[Path] = ARTIFACTS_DIR / "00_meta.json"
VOCAB_PATH: Final[Path] = ARTIFACTS_DIR / "01_vocabulary.csv"
WEIGHTS_PATH: Final[Path] = ARTIFACTS_DIR / "02_model_weights.csv"


def main() -> None:
    """Run inference using saved training artifacts."""
    log_header(LOG, "Inference Demo: Load Artifacts and Generate Text")

    require_artifacts(
        meta_path=META_PATH,
        vocab_path=VOCAB_PATH,
        weights_path=WEIGHTS_PATH,
        train_hint="uv run python src/toy_gpt_train_animals/d_train.py",
    )

    meta: JsonObject = load_meta(META_PATH)
    vocab: ArtifactVocabulary = load_vocabulary_csv(VOCAB_PATH)

    v: int = vocab.vocab_size()
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=v)
    model.weights = load_model_weights_csv(WEIGHTS_PATH, vocab_size=v)

    args: argparse.Namespace = parse_args([])

    LOG.info(
        f"Loaded repo_name={meta.get('repo_name')} model_kind={meta.get('model_kind')}"
    )
    LOG.info(f"Vocab size: {v}")
    LOG.info("Unigram model: predictions are the same regardless of input.")

    # Show predictions (same for any input)
    probs: list[float] = model.forward()
    LOG.info("Top next-token predictions (based on corpus frequency):")
    for tok_id, prob in top_k(probs, k=max(1, args.topk)):
        tok: str | None = vocab.get_id_token(tok_id)
        LOG.info(f"  {tok!r} (ID {tok_id}): {prob:.4f}")

    generated: list[str] = generate_tokens_unigram(
        model=model,
        vocab=vocab,
        num_tokens=max(0, args.num_tokens),
    )

    LOG.info("Generated sequence:")
    LOG.info(f"  {' '.join(generated)}")


if __name__ == "__main__":
    main()
