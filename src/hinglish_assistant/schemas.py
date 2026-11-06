from __future__ import annotations

from pydantic import BaseModel, Field


class AssistRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    rewrite: bool = True
    translate: bool = True


class AssistResponse(BaseModel):
    transliteration: str
    polished_hindi: str | None = None
    english_from_raw: str | None = None
    english_from_polished: str | None = None
    warnings: list[str] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
