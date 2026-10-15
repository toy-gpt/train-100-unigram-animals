"""d_train.py - Training loop module.

Trains the SimpleNextTokenModel on a small token corpus
using unigram (no context - just word frequencies).

A unigram models P(next) - the probability of each word based purely
on how often it appears in the corpus, ignoring all context.

Responsibilities:
- Count token frequencies in the corpus
- Train a single row of weights to predict based on frequency
- Track loss and accuracy per epoch
- Write a CSV log of training progress
- Write inspectable training artifacts (vocabulary, weights, embeddings, meta)

Concepts:
- unigram: predict next token using only corpus frequencies (no context)
- softmax: converts raw scores into probabilities (so predictions sum to 1)
- cross-entropy loss: measures how well predicted probabilities match the correct token
- gradient descent: iterative weight updates to minimize loss

Notes:
- This is intentionally simple: no deep learning framework, no Transformer.
- The model has only ONE row of weights (predictions are context-independent).
- Training updates the same single row for every example.
- token_embeddings.csv is a visualization-friendly projection for levels 100-400;
  in later repos (500+), embeddings become a first-class learned table.
"""

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from toy_gpt_train.c_model import SimpleNextTokenModel
from toy_gpt_train.d_train import (
    make_training_targets,
    row_labeler_unigram,
    train_model,
)
from toy_gpt_train.io_artifacts import (
    write_artifacts,
    write_training_log,
)
from toy_gpt_train.math_training import argmax

from toy_gpt_train_animals.a_tokenizer import DEFAULT_CORPUS_PATH, SimpleTokenizer
from toy_gpt_train_animals.b_vocab import Vocabulary

LOG: logging.Logger = get_logger("TRAIN", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Final[Path] = BASE_DIR / "outputs"
TRAIN_LOG_PATH: Final[Path] = OUTPUTS_DIR / "train_log.csv"


def main() -> None:
    """Run a simple training demo end-to-end."""
    log_header(LOG, "Training Demo: Unigram (Frequency-Based) Model")

    # Step 1: Load and tokenize the corpus.
    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=DEFAULT_CORPUS_PATH)
    tokens: list[str] = tokenizer.get_tokens()

    if not tokens:
        LOG.error("No tokens found. Check corpus file.")
        return

    # Step 2: Build vocabulary (maps tokens <-> integer IDs).
    vocab: Vocabulary = Vocabulary(tokens)

    # Step 3: Convert token strings to integer IDs for training.
    token_ids: list[int] = []
    for tok in tokens:
        tok_id: int | None = vocab.get_token_id(tok)
        if tok_id is None:
            LOG.error(f"Token not found in vocabulary: {tok!r}")
            return
        token_ids.append(tok_id)

    # Step 4: Create training targets (just the tokens themselves for unigram).
    targets: list[int] = make_training_targets(token_ids)
    LOG.info(f"Created {len(targets)} training targets.")

    # Step 5: Initialize model (unigram has only 1 row of weights).
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab.vocab_size())

    # Step 6: Train the model.
    learning_rate: float = 0.1
    epochs: int = 50

    history: list[dict[str, float]] = train_model(
        model=model,
        targets=targets,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    # Step 7: Save training metrics for analysis.
    write_training_log(TRAIN_LOG_PATH, history)

    # Step 7b: Write inspectable artifacts for downstream use.
    write_artifacts(
        base_dir=BASE_DIR,
        corpus_path=DEFAULT_CORPUS_PATH,
        vocab=vocab,
        model=model,
        model_kind="unigram",
        learning_rate=learning_rate,
        epochs=epochs,
        row_labeler=row_labeler_unigram(vocab, vocab.vocab_size()),
    )

    # Step 8: Qualitative check - what does the model predict?
    probs: list[float] = model.forward()
    best_id: int = argmax(probs)
    best_tok: str | None = vocab.get_id_token(best_id)
    LOG.info(
        f"After training, most likely token (based on frequency) "
        f"is {best_tok!r} (ID: {best_id})."
    )


if __name__ == "__main__":
    main()
