from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from hinglish_assistant.client import AssistantClient, ServiceUnavailable  # noqa: E402


st.set_page_config(page_title="Hinglish Smart Writing Assistant", page_icon="✍️", layout="wide")
st.title("✍️ Hinglish Smart Writing Assistant")
st.caption("Roman Hinglish → देवनागरी → natural Hindi → English")

with st.sidebar:
    st.header("Pipeline")
    rewrite = st.toggle("Improve phrasing", value=True)
    translate = st.toggle("Translate to English", value=True)
    st.caption("The app calls your configured Colab inference service. No text is stored by the frontend.")

examples = {
    "Casual update": "kal office se jaldi nikalna hai kyunki ghar pe mehmaan aa rahe hain",
    "College message": "sir assignment kal tak submit kar dunga please thoda time de dijiye",
    "Code-switched": "meeting ke baad Google Drive pe final PPT upload kar dena",
}
example_name = st.selectbox("Try an example", ["Custom input", *examples])
default = examples.get(example_name, "")
text = st.text_area("Write in Roman-script Hinglish", value=default, height=140, max_chars=500,
                    placeholder="e.g., mujhe kal presentation ke liye help chahiye")

if st.button("Improve writing", type="primary", use_container_width=True):
    with st.spinner("Running transliteration and writing models…"):
        try:
            result = AssistantClient.from_environment().assist(text, rewrite=rewrite, translate=translate)
        except (ValueError, ServiceUnavailable) as exc:
            st.error(str(exc))
        else:
            first, second = st.columns(2)
            with first:
                st.subheader("1. Devanagari transliteration")
                st.code(result.transliteration, language=None)
                if result.english_from_raw:
                    st.subheader("3a. English from raw Hindi")
                    st.write(result.english_from_raw)
            with second:
                st.subheader("2. Polished Hindi")
                st.code(result.polished_hindi or "Rewrite disabled", language=None)
                if result.english_from_polished:
                    st.subheader("3b. English from polished Hindi")
                    st.write(result.english_from_polished)
            for warning in result.warnings:
                st.warning(warning)
            with st.expander("Model details"):
                st.json(result.model_versions)

st.divider()
st.caption("Academic prototype using AI4Bharat IndicXlit, IndicTrans2, BharatGen Param-1, and Aksharantar.")
