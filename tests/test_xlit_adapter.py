import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))
from server import extract_xlit_result


def test_extracts_a_string_result() -> None:
    assert extract_xlit_result("नमस्ते") == "नमस्ते"


def test_extracts_top_candidate_from_common_response_shapes() -> None:
    assert extract_xlit_result(["नमस्ते", "नमसते"]) == "नमस्ते"
    assert extract_xlit_result({"hi": ["नमस्ते"]}) == "नमस्ते"
