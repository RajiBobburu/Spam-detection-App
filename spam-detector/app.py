import streamlit as st
import pandas as pd

from utils.loader import load_model
from utils.predictor import predict_text

# ── Page Config ─────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# ── Load Model ─────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

svm_model, tfidf = get_model()

# ── Session State ──────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── UI ─────────────────────────────────────
st.title("🛡️ Spam Detector")
st.caption("Detect whether a message is Spam or Not Spam")

st.markdown("---")

user_input = st.text_area(
    "Enter your message:",
    placeholder="Type your message here...",
    height=150
)

if st.button("🔍 Detect"):
    text = user_input.strip()

    if not text:
        st.warning("⚠️ Please enter some text")
    else:
        label = predict_text(text, svm_model, tfidf)
        is_spam = label == "SPAM"

        if is_spam:
            st.error("🚨 SPAM DETECTED")
        else:
            st.success("✅ NOT SPAM (HAM)")

        # Save history
        st.session_state.history.append({
            "Message": text[:80],
            "Result": label
        })

# ── History ────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.subheader("📋 History")

    df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(df, use_container_width=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()