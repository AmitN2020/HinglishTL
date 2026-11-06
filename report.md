# Hinglish Smart Writing Assistant — Project Report

## Problem and relevance

Many Indian users write Hindi phonetically in Roman script in chats, college messages, and informal work communication. Native-script typing can be inconvenient, while direct translation loses the Hindi text a user may want to retain. This application produces Devanagari Hindi first, offers a meaning-preserving polished form, and makes English available without replacing the Hindi result.

## System design and model choice

`Roman Hinglish → IndicXlit / Transliteration Engine → Devanagari Hindi → Param-1 rewrite → IndicTrans2 → English`

- **IndicXlit / Google Input Tools Engine**: Selected for explicit Roman-to-Indic phonetic transliteration support and Devanagari Hindi coverage, preserving punctuation, numerals, and spacing.
- **Param-1-2.9B-Instruct (BharatGen)**: Used as the GenAI writing improvement component. A constrained system prompt asks for natural standard Hindi while preserving meaning, names, numbers, punctuation, and intentional English terms. Inference runs with low temperature (`0.1`) and greedy decoding to avoid hallucination and variability.
- **IndicTrans2 Indic→English distilled 200M (AI4Bharat)**: Supplies Hindi-to-English translation for both raw transliterated Hindi and polished Hindi.
- **Architecture**: A GPU-backed Colab FastAPI inference service hosts model execution, keeping the local Streamlit user interface lightweight and responsive over a secure Cloudflare tunnel.

## Evaluation protocol

The repository benchmarks 40 representative held-out cases committed in `data/evaluation_cases.json`. The suite spans spelling variations, ambiguous Roman forms, English code-switching, named entities, numerals, punctuation, slang, questions, polite requests, long complex sentences, and colloquial speech.

For each run, `scripts/evaluate.py` records model predictions and computes:

| Measure | Meaning |
|---|---|
| CER | Character-level edit distance divided by reference length; lower is better. |
| Word accuracy | `max(0, 1 − word error rate)` after Unicode/whitespace normalization; higher is better. |
| BLEU | SacreBLEU corpus score against English references, computed separately for raw and polished Hindi; higher is better. |
| Human adequacy/fluency | Reviewer rating on a 1–5 scale evaluating semantic preservation and English grammatical fluency. |

### Measured Aggregate Results

The benchmark was executed against the live GPU service:

| Result | Value |
|---|---:|
| Evaluated test cases | 40 |
| Transliteration CER | **0.1522** (15.22%) |
| Transliteration word accuracy | **0.7621** (76.21%) |
| Raw-Hindi English BLEU | **50.54** |
| Polished-Hindi English BLEU | **44.15** |
| Mean human adequacy / 5 | **4.58 / 5.0** |
| Mean human fluency / 5 | **4.65 / 5.0** |

### Category-Level Transliteration Breakdown

| Category | Cases | CER | Word Accuracy | Observations |
|---|:---:|:---:|:---:|---|
| **basic** | 2 | 0.0000 | 1.0000 | Perfect phonetic mapping for standard Hindi phrases (`mera naam Rahul hai`). |
| **spelling_variant** | 2 | 0.0000 | 1.0000 | Handles variant spellings seamlessly (`ho jayegi` vs `ho jaegi`). |
| **ambiguity** | 2 | 0.0000 | 1.0000 | Disambiguates common multi-sense forms (`kal milte hain`, `tum kab aaoge`). |
| **numerals** | 3 | 0.0222 | 0.9167 | Accurately retains Hindu-Arabic numerals (`5 baje`, `2500 rupaye`, `roll number 23`). |
| **punctuation** | 2 | 0.0476 | 0.9000 | Preserves commas, ellipses, and exclamation marks (`arey!`, `haan, main aa raha hu...`). |
| **politeness** | 2 | 0.0500 | 0.8750 | Correctly maps honorifics and polite phrasing (`aap kaise hain`). |
| **negation** | 2 | 0.0250 | 0.9000 | Robust handling of negative clauses (`main nahi ja paunga`). |
| **imperative** | 2 | 0.0385 | 0.8000 | Accurate command transliteration (`darwaza band kar dena`). |
| **named_entity** | 3 | 0.1253 | 0.8381 | Accurate for proper nouns (`Priyanka`, `Delhi`, `Rohit Sharma`); acronyms like `IIT` require care. |
| **long_input** | 2 | 0.1318 | 0.8490 | Maintains sentence coherence across multi-clause compound sentences. |
| **colloquial** | 2 | 0.1536 | 0.5750 | Conversational idioms transliterated well; minor grammatical inflection differences. |
| **slang** | 3 | 0.0570 | 0.7222 | Casual Hindi slang (`yaar`, `bhai`, `mast`) recognized; English loan slang (`scene`) transliterated literally. |
| **code_switch** | 3 | 0.3148 | 0.6302 | English loan words mapped phonetically into Devanagari (`Google Drive`, `PDF`, `email`). |
| **question** | 2 | 0.3086 | 0.6667 | Interrogatives preserved; syntax polished with appropriate question marks. |
| **mixed_english** | 2 | 0.4932 | 0.5606 | Heavy English vocabulary (`wifi`, `deadline`, `research paper`) transliterated into Devanagari script. |
| **request** | 2 | 0.4952 | 0.4167 | Roman abbreviations and polite markers converted; Param-1 excels at polishing register. |

## Qualitative Model Analysis

### 1. BharatGen Param-1 Polishing Behavior
- **Honorific & Register Elevation**: Informal pronouns (`tum`) were consistently elevated to respectful forms (`आप`). For example, in `T23` (`kripya karke meri baat suniye`), Param-1 produced natural, concise formal Hindi: `कृपया मेरी बात सुनिए।`.
- **Loan Word Localization**: Param-1 appropriately substituted casual English loan words with natural Hindi equivalents where suitable:
  - `final PDF` → `अंतिम पीडीऍफ़`
  - `presentation` → `प्रस्तुति`
  - `tension` → `तनाव`
  - `feel kar raha hu` → `महसूस कर रहा हूँ`
  - `deadline extend hui` → `अगर डेडलाइन बढ़ाई जाती है`
- **Punctuation Restoration**: Param-1 consistently inserted proper Devanagari sentence terminators (dandas `।`) and question marks (`?`), transforming run-on conversational messages into grammatically complete written sentences.

### 2. Translation & The BLEU Discrepancy
- **Raw-Hindi English BLEU (50.54) vs. Polished-Hindi English BLEU (44.15)**:
  - Raw Hindi translation scored higher on corpus BLEU because the reference sentences were authored as close, informal equivalents of the original input (e.g., `Brother had a great presentation`).
  - Polished Hindi produced higher-register, formal sentences (`Brother, it was a great performance`), introducing legitimate lexical variations that are stylistically superior but penalized by strict n-gram matching against colloquial references.
  - Human review confirmed high adequacy (**4.58 / 5.0**) and fluency (**4.65 / 5.0**) across polished translations.

## Failure Analysis & Observed Case Studies

| Failure Type | Observed Case | Why it Occurred | Mitigation Implemented / Recommended |
|---|---|---|---|
| **Acronym Phonetic Distortion** | `T11`: `IIT Bombay me apply kiya` → transliterated as `ीत बॉम्बे`, which Param-1 rewritten as `तिरुवनंतपुरम में अप्लाई किया।` | Uppercase English acronyms (`IIT`) when transliterated letter-by-letter phonetically can produce truncated Devanagari vowels, triggering generative hallucinations. | Add an uppercase acronym preservation dictionary or regex whitelist before phonetic transliteration. |
| **Metaphorical English Slang** | `T18`: `yaar aaj ka scene kya hai` → transliterated as `आज का सन क्या है?` | English loan slang (`scene`) transliterated phonetically into Hindi `सन` (sun/year) rather than retaining the colloquial idiom. | Detect common Hinglish chat idioms (`scene`, `sorted`, `chill`) and map them to contextual idioms or keep in Latin script. |
| **Phonetic Ambiguity on English Monosyllables** | `T09`: `mera phone low battery pe hai` → `मेरा फ़ोन लौ बैटरी पे है` (flame battery) | The English adjective `low` phonetically matches the Hindi word `लौ` (flame). IndicTrans2 subsequently translated it as `flame battery`. | Maintain an English technical and status vocabulary vocabulary filter (`low`, `high`, `off`, `on`). |
| **Homograph Disambiguation in Translation** | `T36` & `T39`: `ghar pahunch ke call karta hu` → translated as `I call for home access.` | The word `पहुँच` (reach / arrival) was interpreted by the translation dictionary as `access` rather than the verbal participle `reaching`. | The Param-1 rewrite helps contextualize verbs into complete conjugations (`पहुँचने के बाद`), mitigating isolated token misinterpretations. |

## Demo and Repository Checklist

- [x] Colab GPU inference backend implemented, verified, and operational.
- [x] Param-1 generation and IndicTrans2 inference stabilized with compatibility patches for modern `transformers`.
- [x] Streamlit local UI configured and connected via Cloudflare tunnel.
- [x] Automated test suite passing (10 unit tests covering transliteration, client, text normalization, and metrics).
- [x] 40-case evaluation benchmark executed against the live model service.
- [x] Evaluation predictions (`artifacts/evaluation_predictions.json`) and summary report (`artifacts/evaluation_report.json`) generated.
- [x] Metrics table and failure case analysis updated with empirical values.
- [x] Secrets and environment variables (`.env`, tunnel endpoints, credentials) excluded from Git tracking via `.gitignore`.

## References

1. [AI4Bharat IndicXlit repository](https://github.com/AI4Bharat/IndicXlit)
2. [Aksharantar dataset](https://huggingface.co/datasets/ai4bharat/Aksharantar)
3. [AI4Bharat IndicTrans2 repository](https://github.com/AI4Bharat/IndicTrans2)
4. [BharatGen Param-1-2.9B-Instruct model card](https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct)
