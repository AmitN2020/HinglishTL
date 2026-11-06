import sys
from pathlib import Path

from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))
from server import extract_xlit_result, transliterate_hindi


def test_extracts_a_string_result() -> None:
    assert extract_xlit_result("नमस्ते") == "नमस्ते"


def test_extracts_top_candidate_from_common_response_shapes() -> None:
    assert extract_xlit_result(["नमस्ते", "नमसते"]) == "नमस्ते"
    assert extract_xlit_result({"hi": ["नमस्ते"]}) == "नमस्ते"


def test_transliterate_hindi_preserves_punctuation_and_numbers() -> None:
    def fake_get(url, params=None, timeout=None):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # If calling Google Input Tools fallback
        if "inputtools.google.com" in url:
            word = params.get("text", "")
            mapping = {"kal": "कल", "milte": "मिलते", "hain": "हैं"}
            mock_resp.json.return_value = ["SUCCESS", [[word, [mapping.get(word, word)]]]]
        else:
            mock_resp.status_code = 404
        return mock_resp

    with patch("requests.get", side_effect=fake_get):
        result = transliterate_hindi("kal, milte hain 123!")
        assert "कल, मिलते हैं 123!" == result
