from __future__ import annotations

import os
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from .schemas import AssistResponse
from .text import validate_roman_hinglish


class ServiceUnavailable(RuntimeError):
    """Raised when the configurable inference service cannot complete a request."""


@dataclass
class AssistantClient:
    base_url: str
    timeout_seconds: int = 90
    api_key: str | None = None

    @classmethod
    def from_environment(cls) -> "AssistantClient":
        load_dotenv()
        return cls(
            base_url=os.getenv("SERVICE_BASE_URL", "").rstrip("/"),
            timeout_seconds=int(os.getenv("SERVICE_TIMEOUT_SECONDS", "90")),
            api_key=os.getenv("SERVICE_API_KEY") or None,
        )

    def assist(self, text: str, rewrite: bool, translate: bool) -> AssistResponse:
        if not self.base_url:
            raise ServiceUnavailable("SERVICE_BASE_URL is not configured. Start the Colab service first.")
        payload = {"text": validate_roman_hinglish(text), "rewrite": rewrite, "translate": translate}
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            response = requests.post(
                f"{self.base_url}/assist", json=payload, headers=headers, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            return AssistResponse.model_validate(response.json())
        except requests.RequestException as exc:
            raise ServiceUnavailable(f"Inference service unavailable: {exc}") from exc
