"""tests/test_train_round_trip.py

REQ: Verify training and inference work end to end on the animals corpus.
WHY: Minimal behavioral check for a repository that reuses the general package.

OBS: Training writes to a temporary folder, never to the repository.
"""

import json
from pathlib import Path

import pytest

from toy_gpt_train_animals import d_train, e_infer
from toy_gpt_train_animals.a_tokenizer import DEFAULT_CORPUS_PATH, SimpleTokenizer


def test_corpus_is_the_animals_corpus() -> None:
    """Test that the local tokenizer reads this repository's animals corpus."""
    assert DEFAULT_CORPUS_PATH.name == "001_animals.txt"
    assert DEFAULT_CORPUS_PATH.is_file()

    tokens: list[str] = SimpleTokenizer(corpus_path=DEFAULT_CORPUS_PATH).get_tokens()
    assert "tabby" in tokens


def test_train_then_infer_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test training into a temporary folder, then inference from those artifacts."""
    artifacts_dir: Path = tmp_path / "artifacts"
    outputs_dir: Path = tmp_path / "outputs"

    # Send everything d_train writes to the temporary folder.
    monkeypatch.setattr(d_train, "BASE_DIR", tmp_path)
    monkeypatch.setattr(d_train, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(d_train, "TRAIN_LOG_PATH", outputs_dir / "train_log.csv")
    d_train.main()

    meta = json.loads((artifacts_dir / "00_meta.json").read_text(encoding="utf-8"))
    assert meta["model_kind"] == "unigram"
    assert meta["corpus"]["filename"] == "001_animals.txt"
    assert (artifacts_dir / "01_vocabulary.csv").is_file()
    assert (artifacts_dir / "02_model_weights.csv").is_file()
    assert (outputs_dir / "train_log.csv").is_file()

    # Point inference at the temporary artifacts.
    monkeypatch.setattr(e_infer, "META_PATH", artifacts_dir / "00_meta.json")
    monkeypatch.setattr(e_infer, "VOCAB_PATH", artifacts_dir / "01_vocabulary.csv")
    monkeypatch.setattr(e_infer, "WEIGHTS_PATH", artifacts_dir / "02_model_weights.csv")
    e_infer.main()
