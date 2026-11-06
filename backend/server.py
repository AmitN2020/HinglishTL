"""FastAPI inference service intended for a GPU-backed Colab session.

Start Param-1 separately with vLLM, then run this server.  The service keeps all
model-specific code off the Streamlit machine and exposes one stable /assist API.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hinglish_assistant.schemas import AssistRequest, AssistResponse  # noqa: E402
from hinglish_assistant.text import normalize_text, validate_roman_hinglish  # noqa: E402

app = FastAPI(title="Hinglish Assistant Inference Service", version="1.0.0")

PARAM_MODEL = os.getenv("PARAM_MODEL_ID", "bharatgenai/Param-1-2.9B-Instruct")
INDICTRANS_MODEL = os.getenv("INDICTRANS_MODEL_ID", "ai4bharat/indictrans2-indic-en-dist-200M")
_xlit = _translator = _tokenizer = _processor = None


def transliterate_hindi(text: str) -> str:
    global _xlit
    if _xlit is None:
        from ai4bharat.transliteration import XlitEngine

        _xlit = XlitEngine(beam_width=4, rescore=True, src_script_type="roman")
    # IndicXlit is word-oriented; its sentence helper preserves spacing/punctuation.
    return normalize_text(_xlit.translit_sentence(text, lang_code="hi"))


def load_translation_model() -> None:
    global _translator, _tokenizer, _processor
    if _translator is not None:
        return
    from IndicTransToolkit import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(INDICTRANS_MODEL, trust_remote_code=True)
    _translator = AutoModelForSeq2SeqLM.from_pretrained(INDICTRANS_MODEL, trust_remote_code=True).cuda()
    _processor = IndicProcessor(inference=True)


def translate_hindi_to_english(text: str) -> str:
    load_translation_model()
    prepared = _processor.preprocess_batch([text], src_lang="hin_Deva", tgt_lang="eng_Latn")
    inputs = _tokenizer(prepared, return_tensors="pt", padding=True, truncation=True).to("cuda")
    generated = _translator.generate(**inputs, num_beams=5, max_new_tokens=256)
    decoded = _tokenizer.batch_decode(generated, skip_special_tokens=True)
    return normalize_text(_processor.postprocess_batch(decoded, lang="eng_Latn")[0])


def rewrite_hindi(text: str) -> str:
    client = OpenAI(base_url=os.getenv("PARAM_API_BASE", "http://127.0.0.1:8000/v1"), api_key="unused")
    message = (
        "Rewrite this Hindi sentence in natural standard Hindi. Preserve its meaning, names, numbers, "
        "punctuation, and intentional English technical terms. Return only the rewritten Hindi sentence.\n\n"
        f"Input: {text}"
    )
    response = client.chat.completions.create(
        model=PARAM_MODEL, messages=[{"role": "user", "content": message}], temperature=0.1, max_tokens=256
    )
    answer = normalize_text(response.choices[0].message.content or "")
    if not answer:
        raise RuntimeError("Param-1 returned an empty rewrite")
    return answer


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "param_model": PARAM_MODEL, "indictrans_model": INDICTRANS_MODEL}


@app.post("/assist", response_model=AssistResponse)
def assist(request: AssistRequest) -> AssistResponse:
    try:
        raw = transliterate_hindi(validate_roman_hinglish(request.text))
        polished = rewrite_hindi(raw) if request.rewrite else None
        return AssistResponse(
            transliteration=raw,
            polished_hindi=polished,
            english_from_raw=translate_hindi_to_english(raw) if request.translate else None,
            english_from_polished=translate_hindi_to_english(polished) if request.translate and polished else None,
            model_versions={"transliteration": "IndicXlit", "rewrite": PARAM_MODEL, "translation": INDICTRANS_MODEL},
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model stage failed: {exc}") from exc
