"""
CIFAR-10 Image Classifier — Streamlit App
Model: cifar10_strong_cnn.keras (ResNet-style CNN)
Input: 32×32 RGB  |  Output: 10-class softmax
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = "cifar10_strong_cnn.keras"

CIFAR10_CLASSES = [
    "✈️ Airplane", "🚗 Automobile", "🐦 Bird", "🐱 Cat",
    "🦌 Deer",     "🐶 Dog",        "🐸 Frog", "🐴 Horse",
    "🚢 Ship",     "🚚 Truck",
]

IMG_SIZE = (32, 32)  # model's expected input resolution

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0d0d0d; }

    .title-block {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .title-block h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        color: #f0f0f0;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    .title-block p {
        color: #888;
        font-size: 0.95rem;
        font-weight: 300;
    }

    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
        text-align: center;
    }
    .result-label {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        color: #e94560;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .confidence-text {
        font-size: 1rem;
        color: #a0a0b0;
        margin-top: 0.3rem;
    }

    .top5-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0;
        border-bottom: 1px solid #222;
        font-size: 0.9rem;
    }
    .top5-row:last-child { border-bottom: none; }
    .top5-label { color: #ccc; }
    .top5-bar-wrap { width: 55%; background: #1e1e1e; border-radius: 4px; height: 8px; }
    .top5-bar { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #e94560, #0f3460); }
    .top5-pct { color: #e94560; font-family: 'Space Mono', monospace; font-size: 0.8rem; width: 3.5rem; text-align: right; }

    div[data-testid="stFileUploader"] {
        border: 2px dashed #333;
        border-radius: 10px;
        padding: 1rem;
    }

    .note {
        font-size: 0.78rem;
        color: #555;
        text-align: center;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Model loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_model() -> tf.keras.Model:
    """Load the Keras model once and cache it across sessions."""
    return tf.keras.models.load_model(MODEL_PATH)


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    """
    Resize to 32×32, convert to RGB, cast to float32.
    The model's Rescaling layer handles /255 normalisation internally.
    Returns shape (1, 32, 32, 3).
    """
    img = image.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # [0, 255] — model rescales internally
    return np.expand_dims(arr, axis=0)       # (1, 32, 32, 3)


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(model: tf.keras.Model, img_array: np.ndarray) -> np.ndarray:
    """Run softmax inference; returns probability vector of shape (10,)."""
    logits = model(img_array, training=False)            # (1, 10)
    probs  = tf.nn.softmax(logits).numpy().squeeze()     # (10,)
    return probs


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🔍 CIFAR-10 Classifier</h1>
    <p>Developed by Md. Tareq Aziz; Phn: +8801701036833</p>
</div>
""", unsafe_allow_html=True)

# Load model upfront — shows spinner only on first load
with st.spinner("Loading model weights…"):
    model = load_model()

st.divider()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="The model resizes your image to 32×32 internally (CIFAR-10 resolution).",
)
st.markdown('<p class="note">Best results: images clearly showing one of — airplane, car, bird, cat, deer, dog, frog, horse, ship, truck</p>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        # Show what the model actually "sees"
        thumb = image.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
        st.caption("Model input (32×32 px):")
        st.image(thumb, width=96)

    with col2:
        with st.spinner("Running inference…"):
            arr   = preprocess(image)
            probs = predict(model, arr)

        top_idx   = int(np.argmax(probs))
        top_label = CIFAR10_CLASSES[top_idx]
        top_conf  = float(probs[top_idx]) * 100

        # ── Primary result card ──────────────────────────────────────────
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">{top_label}</div>
            <div class="confidence-text">Confidence: <strong>{top_conf:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top-5 breakdown ──────────────────────────────────────────────
        st.markdown("#### Top-5 Predictions")
        top5_idx = np.argsort(probs)[::-1][:5]

        for idx in top5_idx:
            label = CIFAR10_CLASSES[idx]
            pct   = float(probs[idx]) * 100
            bar_w = int(pct)  # percentage → pixel width proxy
            st.markdown(f"""
            <div class="top5-row">
                <span class="top5-label">{label}</span>
                <div class="top5-bar-wrap">
                    <div class="top5-bar" style="width:{bar_w}%;"></div>
                </div>
                <span class="top5-pct">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Raw probability table (expandable)
    with st.expander("📊 Full probability distribution"):
        import pandas as pd
        df = pd.DataFrame({
            "Class":       CIFAR10_CLASSES,
            "Probability": [f"{p*100:.4f}%" for p in probs],
            "Raw Score":   [f"{p:.6f}" for p in probs],
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#444; font-size:0.8rem;">'
    'Model: <code>cifar10_strong_cnn.keras</code> · '
    'Framework: TensorFlow/Keras · '
    'Dataset: CIFAR-10 (60 000 images, 10 classes)'
    '</p>',
    unsafe_allow_html=True,
)
