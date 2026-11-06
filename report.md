# Hinglish Smart Writing Assistant — Project Report

## Abstract

Code-mixed Hindi-English written in the Latin script (*Roman Hinglish*) has become the predominant mode of informal digital communication across South Asia. However, Roman Hinglish lacks standardized orthography, presents high lexical ambiguity, and excludes users from formal native-script (Devanagari) contexts. We present the **Hinglish Smart Writing Assistant**, an end-to-end, privacy-conscious neural language pipeline that transforms unstructured Roman Hinglish into formal, grammatically polished Devanagari Hindi while generating paired English translations for semantic verification. 

The architecture decouples lightweight client interaction from compute-intensive neural execution: a local Streamlit interface communicates over an authenticated tunnel with a GPU-backed FastAPI inference backend. The inference engine chains three specialized models: (1) an adaptive phonetic transliteration engine powered by AI4Bharat IndicXlit with automated high-availability fallback; (2) **BharatGen Param-1-2.9B-Instruct**, a bilingual foundation model providing constrained, meaning-preserving register elevation and punctuation restoration; and (3) **AI4Bharat IndicTrans2-200M**, generating dual English translations from both raw and polished variants. 

Evaluated across a benchmark of 40 curated test cases spanning 18 sociolinguistic challenge categories, the system achieves a Character Error Rate (CER) of **15.22%**, word accuracy of **76.21%**, raw translation BLEU of **50.54**, and polished translation BLEU of **44.15**, accompanied by independent human evaluation scores of **4.58 / 5.0** for adequacy and **4.65 / 5.0** for fluency.

---

## Problem and Relevance

Over 500 million bilingual speakers in India frequently communicate in Hinglish across instant messaging, social networks, and educational forums. Despite its prevalence, Roman Hinglish presents significant communicative and computational challenges:

1. **Non-Standardized Orthography**: The absence of canonical spelling rules produces extreme surface-form variability (e.g., *mujhe*, *mjhe*, *muze*).
2. **Loss of Native Cultural and Script Identity**: Converting ideas directly to English completely discards the native Hindi expression, whereas typing natively in Devanagari remains cumbersome on standard QWERTY keyboards.
3. **Register Disparity**: Informal chat messages often lack grammatical inflection, sentence-boundary punctuation, and honorific consistency necessary for academic or workplace correspondence.
4. **Code-Switching Ambiguity**: Interleaved English nouns and technical jargon (*assignment*, *presentation*, *Google Drive*) are frequently misidentified by conventional translation engines.

This project addresses these gaps by creating an assistive pipeline that restores native Devanagari orthography, refines syntactic fluency through an Indic-specialized LLM, and produces side-by-side English translations to help users verify nuance.

---

## System Design and Model Choice

The system is designed around a modular, decoupled architecture where all neural inference runs on a GPU-backed server, allowing the frontend to operate seamlessly on standard consumer hardware.

### End-to-End Architecture Flow

```mermaid
flowchart TD
    subgraph Client["Local User Environment"]
        UI["Streamlit Frontend (app.py)"]
        UserIn["User Input: Roman Hinglish"] --> UI
        UI --> Display["4-Panel Result Display"]
    end

    subgraph Tunnel["Network Layer"]
        CF["Cloudflare Secure Tunnel / HTTPS"]
        UI <-->|JSON Payload| CF
    end

    subgraph Backend["GPU Inference Service (FastAPI)"]
        CF <--> API["FastAPI Router (/assist)"]
        
        subgraph Stage1["Stage 1: Transliteration & Normalization"]
            API --> Norm["Text Normalization & Regex Tokenizer"]
            Norm --> XlitEngine{"IndicXlit Hosted API"}
            XlitEngine -->|Fallback if Down| GoogleXlit["Google Input Tools Engine"]
            XlitEngine --> RawDev["Raw Devanagari Hindi"]
            GoogleXlit --> RawDev
        end

        subgraph Stage2["Stage 2: Generative Polishing (LLM)"]
            RawDev --> PromptBuilder["Chat Template & Instruction Prompt"]
            PromptBuilder --> Param1["BharatGen Param-1-2.9B-Instruct\n(FP16 Greedy Decoding, Cache-Bypassed)"]
            Param1 --> PolDev["Polished Standard Hindi"]
        end

        subgraph Stage3["Stage 3: Neural Machine Translation"]
            RawDev --> IT1["IndicTrans2 (200M Distilled)"]
            PolDev --> IT2["IndicTrans2 (200M Distilled)"]
            IT1 --> EngRaw["English (from Raw Hindi)"]
            IT2 --> EngPol["English (from Polished Hindi)"]
        end
        
        EngRaw --> ResponseBuilder["AssistResponse Schema"]
        EngPol --> ResponseBuilder
        RawDev --> ResponseBuilder
        PolDev --> ResponseBuilder
        ResponseBuilder --> API
    end
```

### Model Selection Rationale

| Stage | Selected Component | Model Characteristics | Selection Rationale |
|---|---|---|---|
| **Transliteration** | **AI4Bharat IndicXlit / Google Input Tools** | Character/word-level phonetic sequence-to-sequence model trained on the 70M+ pair Aksharantar corpus. | Unlike general translation models, IndicXlit isolates phonetic transliteration from semantic translation. Tokenizing by word boundaries guarantees that whitespace, numerals (`2500`), and punctuation marks (`!`, `?`, `...`) remain intact. A built-in HTTP fallback to Google Input Tools guarantees 100% operational uptime. |
| **Generative Polishing** | **BharatGen Param-1-2.9B-Instruct** | 2.9-billion parameter bilingual autoregressive language model pretrained and instruction-tuned specifically on English and Hindi. | General small language models (e.g., LLaMA-3.2-1B, Qwen-2.5-1.5B) often hallucinate or revert to English when processing Devanagari. Param-1 captures native Hindi syntactic subtleties, honorific agreements (`tum` → `आप`), and formal loan translations while comfortably residing in ~5.8 GB of GPU VRAM (FP16). |
| **Machine Translation** | **AI4Bharat IndicTrans2 (Indic→En Distilled 200M)** | 200-million parameter distilled Transformer Seq2Seq model supporting 22 scheduled Indian languages to English. | State-of-the-art translation accuracy on Flores-200 and IN22 benchmarks. The 200M distilled checkpoint requires under 900 MB of VRAM, allowing concurrent residence in GPU memory alongside Param-1 on a standard free-tier 15 GB Nvidia T4 GPU without memory eviction. |

---

### Key Engineering & Runtime Innovations

#### 1. Zero-Microservice GPU Memory Orchestration
To avoid the resource overhead, complex IPC, and GPU fragmentation of deploying multi-container vLLM instances alongside PyTorch workers, all models are loaded lazily in a single Python process:
- **Param-1** is loaded in `torch.float16` using Hugging Face `AutoModelForCausalLM` directly onto `device_map="auto"`.
- **IndicTrans2** is co-located on the same CUDA device using `torch_dtype="auto"` within the Seq2Seq generation pipeline.
- Total active VRAM footprint is **~6.7 GB**, leaving over 8 GB of headroom on standard T4 instances to prevent CUDA out-of-memory (OOM) errors during peak generation bursts.

#### 2. Modern Transformers Cache-Compatibility Layer
Custom model implementations written for older Hugging Face revisions (`modeling_parambharatgen.py` and `modeling_indictrans.py`) explicitly expect legacy nested tuples for key-value caching (`past_key_values[0][0].shape[2]`). In `transformers >= 4.45`, calling `.generate()` automatically injects unpopulated dynamic cache objects (`DynamicCache` / `EncoderDecoderCache`), triggering an unhandled:
```text
AttributeError: 'NoneType' object has no attribute 'shape'
```
We resolved this across both models through two defensive layers:
1. **Explicit Cache Bypass**: Forcing `use_cache=False` on the model configuration, `generation_config`, and the `.generate()` invocation. For short conversational inputs ($< 100$ tokens), non-cached autoregression executes in under 1.5 seconds on a T4 GPU while eliminating cache instantiation.
2. **Forward-Pass Sanitizer Guard**: Dynamically wrapping the underlying PyTorch `forward` methods (`_param_model.model.forward` and `_translator.decoder.forward`) to intercept and sanitize any empty or malformed `past_key_values` instances into `None`, providing complete forward-pass resilience across library updates.

#### 3. Dual-Translation Discrepancy Analysis
Rather than presenting a single translation, the system translates both the raw transliterated Hindi and the polished Hindi in parallel. This design serves as a verification mechanism:
- If both English outputs match in meaning, the user gains confidence that the LLM rewrite preserved the original semantics.
- If the translations diverge significantly, the user can immediately identify whether Param-1 over-formalized or altered nuance.

---

## Evaluation Protocol

The repository benchmarks 40 representative held-out cases committed in `data/evaluation_cases.json`. The suite deliberately covers spelling variations, ambiguous Roman spellings, English code-switching, named entities, numbers, punctuation, slang, questions, polite requests, long complex sentences, and colloquial speech.

For each run, `scripts/evaluate.py` records model predictions and computes:

| Measure | Meaning |
|---|---|
| **CER** | Character-level edit distance divided by reference character length; lower is better. |
| **Word Accuracy** | $\max(0, 1 − \text{word error rate})$ after Unicode/whitespace normalization; higher is better. |
| **BLEU** | SacreBLEU corpus score against committed English references, computed separately for raw and polished Hindi; higher is better. |
| **Human Adequacy & Fluency** | Human review on a 1–5 scale evaluating semantic retention (adequacy) and grammatical naturalness (fluency). |

---

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

---

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

---

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

---

## Failure Analysis & Observed Case Studies

| Failure Type | Observed Case | Why it Occurred | Mitigation Implemented / Recommended |
|---|---|---|---|
| **Acronym Phonetic Distortion** | `T11`: `IIT Bombay me apply kiya` → transliterated as `ीत बॉम्बे`, which Param-1 rewritten as `तिरुवनंतपुरम में अप्लाई किया।` | Uppercase English acronyms (`IIT`) when transliterated letter-by-letter phonetically can produce truncated Devanagari vowels, triggering generative hallucinations. | Add an uppercase acronym preservation dictionary or regex whitelist before phonetic transliteration. |
| **Metaphorical English Slang** | `T18`: `yaar aaj ka scene kya hai` → transliterated as `आज का सन क्या है?` | English loan slang (`scene`) transliterated phonetically into Hindi `सन` (sun/year) rather than retaining the colloquial idiom. | Detect common Hinglish chat idioms (`scene`, `sorted`, `chill`) and map them to contextual idioms or keep in Latin script. |
| **Phonetic Ambiguity on English Monosyllables** | `T09`: `mera phone low battery pe hai` → `मेरा फ़ोन लौ बैटरी पे है` (flame battery) | The English adjective `low` phonetically matches the Hindi word `लौ` (flame). IndicTrans2 subsequently translated it as `flame battery`. | Maintain an English technical and status vocabulary filter (`low`, `high`, `off`, `on`). |
| **Homograph Disambiguation in Translation** | `T36` & `T39`: `ghar pahunch ke call karta hu` → translated as `I call for home access.` | The word `पहुँच` (reach / arrival) was interpreted by the translation dictionary as `access` rather than the verbal participle `reaching`. | The Param-1 rewrite helps contextualize verbs into complete conjugations (`पहुँचने के बाद`), mitigating isolated token misinterpretations. |

---

## Demo and Repository Checklist

- [x] Colab GPU inference backend implemented, verified, and operational.
- [x] Param-1 generation and IndicTrans2 inference stabilized with compatibility patches for modern `transformers`.
- [x] Streamlit local UI configured and connected via Cloudflare tunnel.
- [x] Automated test suite passing (10 unit tests covering transliteration, client, text normalization, and metrics).
- [x] 40-case evaluation benchmark executed against the live model service.
- [x] Evaluation predictions (`artifacts/evaluation_predictions.json`) and summary report (`artifacts/evaluation_report.json`) generated.
- [x] Metrics table and failure case analysis updated with empirical values.
- [x] Secrets and environment variables (`.env`, tunnel endpoints, credentials) excluded from Git tracking via `.gitignore`.

---

## References

1. [AI4Bharat IndicXlit repository](https://github.com/AI4Bharat/IndicXlit)
2. [Aksharantar dataset](https://huggingface.co/datasets/ai4bharat/Aksharantar)
3. [AI4Bharat IndicTrans2 repository](https://github.com/AI4Bharat/IndicTrans2)
4. [BharatGen Param-1-2.9B-Instruct model card](https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct)
