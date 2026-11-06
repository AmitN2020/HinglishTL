import pytest

from hinglish_assistant.text import normalize_text, validate_roman_hinglish


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  kal\n milte\t hain ") == "kal milte hain"


def test_validate_text_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        validate_roman_hinglish("   ")


def test_validate_text_rejects_oversize_input() -> None:
    with pytest.raises(ValueError):
        validate_roman_hinglish("a" * 501)
