"""FastAPI inference service intended for a GPU-backed Colab session.

All three models load lazily in this process. Param-1 uses its Transformers
implementation directly because its custom architecture is not vLLM-compatible.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
import requests

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hinglish_assistant.schemas import AssistRequest, AssistResponse  # noqa: E402
from hinglish_assistant.text import normalize_text, validate_roman_hinglish  # noqa: E402

app = FastAPI(title="Hinglish Assistant Inference Service", version="1.0.0")

PARAM_MODEL = os.getenv("PARAM_MODEL_ID", "bharatgenai/Param-1-2.9B-Instruct")
INDICTRANS_MODEL = os.getenv("INDICTRANS_MODEL_ID", "ai4bharat/indictrans2-indic-en-dist-200M")
XLIT_API_BASE_URL = os.getenv("XLIT_API_BASE_URL", "https://xlit-api.ai4bharat.org").rstrip("/")
_translator = _tokenizer = _processor = None
_param_model = _param_tokenizer = None
_ROMAN_WORD = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")


def extract_xlit_result(result: object) -> str:
    """Extract IndicXlit's top Hindi candidate across API response variants."""
    if isinstance(result, dict):
        result = result.get("hi") or result.get("hin") or result.get("result")
    if isinstance(result, list):
        result = result[0] if result else ""
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError("IndicXlit returned no Hindi candidate")
    return result.strip()


def transliterate_word(word: str) -> str:
    """Call AI4Bharat's hosted, word-oriented IndicXlit endpoint."""
    try:
        response = requests.get(
            f"{XLIT_API_BASE_URL}/tl/hi/{quote(word, safe='')}", timeout=20
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"IndicXlit hosted API is unavailable: {exc}") from exc
    if not payload.get("success", True):
        raise RuntimeError(f"IndicXlit rejected '{word}': {payload.get('error', 'unknown error')}")
    return extract_xlit_result(payload.get("result"))


def transliterate_hindi(text: str) -> str:
    """Transliterate Roman word tokens while preserving spaces, numbers, and punctuation."""
    return normalize_text(_ROMAN_WORD.sub(lambda match: transliterate_word(match.group(0)), text))


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


def load_param_model() -> None:
    """Load the custom Param-1 architecture through Transformers, not vLLM."""
    global _param_model, _param_tokenizer
    if _param_model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Param-1 requires a CUDA GPU. Select a GPU runtime in Colab.")
    _param_tokenizer = AutoTokenizer.from_pretrained(PARAM_MODEL, trust_remote_code=False)
    _param_model = AutoModelForCausalLM.from_pretrained(
        PARAM_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()


def rewrite_hindi(text: str) -> str:
    load_param_model()
    import torch

    conversation = [
        {
            "role": "system",
            "content": (
                "Rewrite Hindi into natural standard Hindi. Preserve meaning, names, numbers, punctuation, "
                "and intentional English technical terms. Return only the rewritten Hindi sentence."
            ),
        },
        {"role": "user", "content": text},
    ]
    inputs = _param_tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True
    ).to(_param_model.device)
    with torch.inference_mode():
        output = _param_model.generate(
            inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=_param_tokenizer.eos_token_id,
            use_cache=True,
        )
    generated_tokens = output[0][inputs.shape[-1] :]
    answer = normalize_text(_param_tokenizer.decode(generated_tokens, skip_special_tokens=True))
    if not answer:
        raise RuntimeError("Param-1 returned an empty rewrite")
    return answer


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "param_model": PARAM_MODEL,
        "indictrans_model": INDICTRANS_MODEL,
        "transliteration": "AI4Bharat IndicXlit hosted API",
    }


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
            model_versions={
                "transliteration": "AI4Bharat IndicXlit (hosted API)",
                "rewrite": PARAM_MODEL,
                "translation": INDICTRANS_MODEL,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model stage failed: {exc}") from exc
