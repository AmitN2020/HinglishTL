# Hinglish Smart Writing Assistant — Project Report

## Problem and relevance

Many Indian users write Hindi phonetically in Roman script in chats, college messages, and informal work communication. Native-script typing can be inconvenient, while direct translation loses the Hindi text a user may want to retain. This application produces Devanagari Hindi first, offers a meaning-preserving polished form, and makes English available without replacing the Hindi result.

## System design and model choice

`Roman Hinglish → IndicXlit → Devanagari Hindi → Param-1 rewrite → IndicTrans2 → English`

- **IndicXlit** is selected for its explicit Roman-to-Indic transliteration support and Hindi coverage. It was trained using Aksharantar, a large Indic transliteration resource.
- **Param-1-2.9B-Instruct** is used as the GenAI component. A constrained prompt asks for natural standard Hindi while preserving meaning, names, numbers, punctuation, and intentional English terms. Low temperature reduces rewrite variability.
- **IndicTrans2 Indic→English distilled 200M** supplies practical Hindi-to-English translation for the course demo.
- A Colab GPU service hosts models; the Streamlit frontend is usable locally through one configurable API URL.

## Evaluation protocol

The repository commits 40 representative held-out cases in `data/evaluation_cases.json`. The suite deliberately includes spelling variants, ambiguous Roman forms, code-switching, named entities, numbers, punctuation, slang, questions, requests, long inputs, and colloquial speech.

For each run, `scripts/evaluate.py` records model outputs and reports:

| Measure | Meaning |
|---|---|
| CER | Character-level edit distance divided by reference length; lower is better. |
| Word accuracy | `max(0, 1 − word error rate)` after Unicode/whitespace normalization; higher is better. |
| BLEU | Corpus BLEU against committed English references, separately for raw and polished Hindi; higher is better. |
| Human adequacy/fluency | A reviewer assigns 1–5 scores and a short note for each English output. |

Paste the generated aggregate values here after the final model run:

| Result | Value |
|---|---:|
| Transliteration CER | _run pending_ |
| Transliteration word accuracy | _run pending_ |
| Raw-Hindi English BLEU | _run pending_ |
| Polished-Hindi English BLEU | _run pending_ |
| Mean human adequacy / 5 | _run pending_ |
| Mean human fluency / 5 | _run pending_ |

## Expected failure analysis

| Failure type | Why it occurs | Mitigation |
|---|---|---|
| Ambiguous Roman spellings | Roman Hindi does not uniquely encode vowels, aspiration, or word boundaries. | Show the source alongside output; later add n-best alternatives and user correction. |
| Named entities and English terms | Transliteration may convert brands/acronyms when users expect them unchanged. | Preserve a protected-token list and compare raw/polished output. |
| Informal slang | A standard-Hindi rewrite can over-formalize intent or tone. | Make rewrite optional and constrain the prompt to preserve register when requested. |
| Meaning drift in polishing | A generative model may add, remove, or soften details. | Use low temperature, output-only prompt instructions, and human adequacy review. |
| Translation of code-switching | Mixed Hindi/English tokens may be translated inconsistently. | Evaluate raw and polished translation separately; retain English technical tokens in the rewrite prompt. |
| Tunnel or GPU unavailability | The frontend depends on the temporary Colab service. | Use `/health`, clear UI errors, and record a demo video plus screenshots. |

## Demo and repository checklist

- [ ] Add screenshots of the Streamlit UI to `assets/` and link them here.
- [ ] Record a short end-to-end demo video and link it in the README.
- [ ] Run `python scripts/evaluate.py`; commit the final metrics report and reviewed predictions if permitted.
- [ ] Replace pending metrics above with measured values.
- [ ] Push this repository to GitHub with no `.env`, tunnel URLs, tokens, or downloaded weights.

## References

1. [AI4Bharat IndicXlit repository](https://github.com/AI4Bharat/IndicXlit)
2. [Aksharantar dataset](https://huggingface.co/datasets/ai4bharat/Aksharantar)
3. [AI4Bharat IndicTrans2 repository](https://github.com/AI4Bharat/IndicTrans2)
4. [BharatGen Param-1-2.9B-Instruct model card](https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct)
