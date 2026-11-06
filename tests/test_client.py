import pytest

from hinglish_assistant.client import AssistantClient, ServiceUnavailable


def test_missing_endpoint_is_actionable() -> None:
    with pytest.raises(ServiceUnavailable, match="SERVICE_BASE_URL"):
        AssistantClient("").assist("namaste", rewrite=True, translate=True)
