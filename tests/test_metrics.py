import pytest

from hinglish_assistant.metrics import character_error_rate, corpus_bleu, word_accuracy


def test_exact_text_has_perfect_scores() -> None:
    assert character_error_rate("मेरा नाम राहुल है", "मेरा नाम राहुल है") == 0
    assert word_accuracy("मेरा नाम राहुल है", "मेरा नाम राहुल है") == 1


def test_metrics_normalize_whitespace() -> None:
    assert character_error_rate("मेरा   नाम", "मेरा नाम") == 0
    assert word_accuracy("मेरा नाम", "मेरा नाम है") < 1


def test_bleu_rewards_exact_prediction() -> None:
    assert corpus_bleu(["I am going home."], ["I am going home."]) == pytest.approx(100.0)
