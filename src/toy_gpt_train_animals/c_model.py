"""c_model.py - Simple model module.

Defines a minimal next-token prediction model for unigram (no context).
  A unigram models P(next) - just word frequencies, ignoring all context.

Responsibilities:
- Represent a simple parameterized model that outputs the same
  probability distribution regardless of input.
- Convert scores into probabilities using softmax.
- Provide a forward pass (no training in this module).

This model is intentionally simple:
- one weight vector (1D: just next_token scores)
- one forward computation that ignores input
- no learning here

Training is handled in a different module.
"""

import logging

from datafun_toolkit.logger import get_logger, log_header
from toy_gpt_train.c_model import SimpleNextTokenModel

__all__ = ["SimpleNextTokenModel"]

LOG: logging.Logger = get_logger("MODEL", level="INFO")


def main() -> None:
    """Demonstrate a forward pass of the simple unigram model."""
    # Local imports keep modules decoupled.
    from toy_gpt_train_animals.a_tokenizer import DEFAULT_CORPUS_PATH, SimpleTokenizer
    from toy_gpt_train_animals.b_vocab import Vocabulary

    log_header(LOG, "Simple Next-Token Model Demo")

    # Step 1: Tokenize input text.
    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=DEFAULT_CORPUS_PATH)
    tokens: list[str] = tokenizer.get_tokens()

    if not tokens:
        LOG.info("No tokens available for demonstration.")
        return

    # Step 2: Build vocabulary.
    vocab: Vocabulary = Vocabulary(tokens)

    # Step 3: Initialize model.
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab.vocab_size())

    # Step 4: Forward pass (unigram ignores input).
    probs: list[float] = model.forward()

    # Step 5: Inspect results.
    LOG.info("Unigram ignores input - same predictions for any context:")
    LOG.info("Output probabilities for next token:")
    for idx, prob in enumerate(probs):
        tok: str | None = vocab.get_id_token(idx)
        LOG.info(f"  {tok!r} (ID {idx}) -> {prob:.4f}")


if __name__ == "__main__":
    main()
