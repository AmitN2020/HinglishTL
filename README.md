# Hinglish Smart Writing Assistant

A college LLM-course mini project that turns Roman-script Hinglish into Hindi Devanagari, improves phrasing with an Indian-language LLM, and optionally translates both raw and polished Hindi into English.

## What it demonstrates

| Stage | Model | Purpose |
|---|---|---|
| Transliteration | AI4Bharat IndicXlit | Roman Hinglish → Devanagari Hindi |
| Writing improvement | BharatGen Param-1-2.9B-Instruct | Meaning-preserving natural Hindi rewrite |
| Translation | AI4Bharat IndicTrans2 Indic→English distilled 200M | Hindi → English |

The application is deliberately split into a local Streamlit UI and a GPU-backed Colab FastAPI service. This keeps the demo laptop light while still using the stated pretrained models end-to-end. Transliteration calls AI4Bharat's hosted IndicXlit API; Param-1 and IndicTrans2 run in Colab.

## Quick start

1. Create a Python 3.12 virtual environment and install UI dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`, set `SERVICE_BASE_URL` to the temporary public URL for your running Colab service, and load it into your shell.
3. Run the frontend:

   ```bash
   streamlit run app.py
   ```

The UI includes examples and shows raw transliteration, polished Hindi, and separate English translations for both variants.

## Colab inference service

Use a GPU runtime and accept the access conditions for the gated [IndicTrans2 checkpoint](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) while logged into Hugging Face. The reproducible path is the [Colab notebook](notebooks/hinglish_assistant_colab.ipynb), which installs, authenticates, starts, warms up, and tunnels the service.

To start it manually, install `backend/requirements.txt`, clone this repository, then start the single inference service:

```bash
pip install -r backend/requirements.txt
uvicorn backend.server:app --host 0.0.0.0 --port 7860
```

Expose port 7860 with the tunnel permitted by your Colab environment. Put its HTTPS URL in `SERVICE_BASE_URL`. Confirm `GET /health` before running the Streamlit UI. Tunnel URLs and any tokens are secrets: keep them out of Git.

> The service calls AI4Bharat's hosted IndicXlit API for transliteration, avoiding the legacy local package stack. It loads Param-1 directly through Transformers. The first request downloads and loads the models, so allow several minutes before expecting a response. The default IndicTrans2 checkpoint is the smaller distilled Indic→English model, chosen for a practical course demo.

## Evaluation

The committed `data/evaluation_cases.json` contains 40 representative Hinglish sentences spanning spelling variation, ambiguity, English code-switches, names, numerals, punctuation, slang, long inputs, and rewrite-sensitive requests.

Run the benchmark against the live service:

```bash
python scripts/evaluate.py
```

It writes predictions and a report under `artifacts/` with Character Error Rate (CER), word accuracy, and English BLEU scores.

### Empirical Benchmark Results

| Metric | Measured Result |
|---|---:|
| Evaluated Test Cases | 40 |
| Transliteration CER | **15.22%** |
| Transliteration Word Accuracy | **76.21%** |
| Raw-Hindi English BLEU | **50.54** |
| Polished-Hindi English BLEU | **44.15** |

Run automated tests with `PYTHONPATH=src pytest -q`.

## Demo checklist

1. Start the Colab API and verify `/health`.
2. Start Streamlit and enter a code-switched, informal sentence.
3. Toggle rewrite and translation, then compare raw and polished English outputs.
4. Run the 40-case evaluator and open the generated report.
5. Show the failure analysis and screenshots in [`report.md`](report.md).

## References

- [IndicXlit and Aksharantar](https://github.com/AI4Bharat/IndicXlit)
- [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)
- [Param-1-2.9B-Instruct](https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct)

See `report.md` for project framing, evaluation protocol, and the submission checklist.
