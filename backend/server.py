"""FastAPI inference service intended for a GPU-backed Colab session.

All models load lazily in this process. Param-1 uses its Transformers
implementation directly because its custom architecture is not vLLM-compatible.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import traceback
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
import requests

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hinglish_assistant.schemas import AssistRequest, AssistResponse  # noqa: E402
from hinglish_assistant.text import normalize_text, validate_roman_hinglish  # noqa: E402

logger = logging.getLogger("hinglish_assistant_backend")
app = FastAPI(title="Hinglish Assistant Inference Service", version="1.0.0")

PARAM_MODEL = os.getenv("PARAM_MODEL_ID", "bharatgenai/Param-1-2.9B-Instruct")
INDICTRANS_MODEL = os.getenv("INDICTRANS_MODEL_ID", "ai4bharat/indictrans2-indic-en-dist-200M")
XLIT_API_BASE_URL = os.getenv("XLIT_API_BASE_URL", "https://xlit-api.ai4bharat.org").rstrip("/")

_translator = _tokenizer = _processor = None
_param_model = _param_tokenizer = None
_xlit_engine = None
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
    """Transliterate a single Roman Hinglish word to Hindi Devanagari.

    Tries AI4Bharat's hosted endpoint first, then falls back to Google Input Tools API
    if the AI4Bharat service is unreachable or down.
    """
    global _xlit_engine

    # If ai4bharat-transliteration package is installed and usable, prefer it
    if _xlit_engine is not False:
        if _xlit_engine is None:
            try:
                from ai4bharat.transliteration import XlitEngine

                _xlit_engine = XlitEngine("hi", beam_width=4, rescore=True)
            except Exception:
                _xlit_engine = False
        if _xlit_engine:
            try:
                res = _xlit_engine.translit_word(word, topk=1)
                return extract_xlit_result(res)
            except Exception as exc:
                logger.warning("Local XlitEngine failed for '%s': %s", word, exc)

    # Try hosted IndicXlit API
    try:
        response = requests.get(
            f"{XLIT_API_BASE_URL}/tl/hi/{quote(word, safe='')}", timeout=4
        )
        if response.status_code == 200:
            payload = response.json()
            if payload.get("success", True) and payload.get("result"):
                return extract_xlit_result(payload.get("result"))
    except Exception:
        pass

    # Reliable fallback: Google Input Tools API (hi-t-i0-und)
    try:
        response = requests.get(
            "https://inputtools.google.com/request",
            params={"text": word, "itc": "hi-t-i0-und", "num": "1"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        if data and data[0] == "SUCCESS" and len(data) > 1 and data[1]:
            candidates = data[1][0][1]
            if candidates and isinstance(candidates[0], str) and candidates[0].strip():
                return candidates[0].strip()
    except Exception as exc:
        logger.warning("Transliteration fallback failed for '%s': %s", word, exc)

    return word


def transliterate_hindi(text: str) -> str:
    """Transliterate Roman word tokens while preserving spaces, numbers, and punctuation."""
    return normalize_text(_ROMAN_WORD.sub(lambda match: transliterate_word(match.group(0)), text))


def load_translation_model() -> None:
    global _translator, _tokenizer, _processor
    if _translator is not None:
        return

    try:
        from IndicTransToolkit.processor import IndicProcessor
    except ImportError:
        from IndicTransToolkit import IndicProcessor  # type: ignore[no-redef]
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logger.info("Loading translation model: %s", INDICTRANS_MODEL)
    _tokenizer = AutoTokenizer.from_pretrained(INDICTRANS_MODEL, trust_remote_code=True)
    _translator = AutoModelForSeq2SeqLM.from_pretrained(
        INDICTRANS_MODEL, trust_remote_code=True, torch_dtype="auto"
    ).cuda()
    _processor = IndicProcessor(inference=True)


def translate_hindi_to_english(text: str) -> str:
    load_translation_model()
    prepared = _processor.preprocess_batch([text], src_lang="hin_Deva", tgt_lang="eng_Latn")
    inputs = _tokenizer(prepared, return_tensors="pt", padding=True, truncation=True).to("cuda")
    generated = _translator.generate(**inputs, num_beams=5, max_new_tokens=256)
    decoded = _tokenizer.batch_decode(generated, skip_special_tokens=True)
    return normalize_text(_processor.postprocess_batch(decoded, lang="eng_Latn")[0])


def load_param_model() -> None:
    """Load the custom Param-1 architecture through Transformers directly on GPU."""
    global _param_model, _param_tokenizer
    if _param_model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Param-1 requires a CUDA GPU. Select a GPU runtime in Colab.")

    logger.info("Loading Param-1 model: %s", PARAM_MODEL)
    _param_tokenizer = AutoTokenizer.from_pretrained(PARAM_MODEL, trust_remote_code=False)
    if _param_tokenizer.pad_token_id is None:
        _param_tokenizer.pad_token_id = _param_tokenizer.eos_token_id

    _param_model = AutoModelForCausalLM.from_pretrained(
        PARAM_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    # The custom Param-1 architecture uses legacy KV tuples incompatible with
    # modern transformers DynamicCache (which causes AttributeError: 'NoneType' object has no attribute 'shape').
    _param_model.config.use_cache = False
    if hasattr(_param_model, "generation_config") and _param_model.generation_config is not None:
        _param_model.generation_config.use_cache = False
    _param_model._supports_cache_class = False

    # Guard monkey-patch: sanitize past_key_values if any is passed with None elements
    orig_forward = _param_model.model.forward

    def safe_forward(*args, **kwargs):
        if "past_key_values" in kwargs:
            pkv = kwargs["past_key_values"]
            if pkv is not None:
                try:
                    if len(pkv) == 0 or pkv[0] is None or pkv[0][0] is None:
                        kwargs["past_key_values"] = None
                except Exception:
                    kwargs["past_key_values"] = None
        elif len(args) > 3:
            pkv = args[3]
            if pkv is not None:
                try:
                    if len(pkv) == 0 or pkv[0] is None or pkv[0][0] is None:
                        args = args[:3] + (None,) + args[4:]
                except Exception:
                    args = args[:3] + (None,) + args[4:]
        return orig_forward(*args, **kwargs)

    _param_model.model.forward = safe_forward


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
    chat_text = _param_tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    model_inputs = _param_tokenizer([chat_text], return_tensors="pt").to(_param_model.device)
    with torch.inference_mode():
        output = _param_model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=_param_tokenizer.eos_token_id,
            pad_token_id=_param_tokenizer.pad_token_id or _param_tokenizer.eos_token_id,
            use_cache=False,
        )
    input_len = model_inputs["input_ids"].shape[-1]
    generated_tokens = output[0][input_len:]
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
        "transliteration": "IndicXlit / Google Input Tools fallback",
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
                "transliteration": "AI4Bharat IndicXlit / Google Input Tools",
                "rewrite": PARAM_MODEL,
                "translation": INDICTRANS_MODEL,
            },
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Model stage failed: {exc}") from exc
