"""
AI Playground — Project 2: Image Recognition
Streamlit app version.

Run locally:
    streamlit run app.py
"""

import os
import urllib.request

import pandas as pd
import streamlit as st
from PIL import Image

from model import load_model, predict_image

st.set_page_config(
    page_title="Image Recognition",
    page_icon="🖼️",
    layout="centered",
)

FALLBACK_IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg"
FALLBACK_IMAGE_PATH = "sample_image.jpg"


@st.cache_resource(show_spinner="Loading MobileNetV2 (pretrained on ImageNet)...")
def get_model():
    return load_model()


model = get_model()

st.title("🖼️ Image Recognition")
st.caption("MobileNetV2, pretrained on ImageNet — 1,000 object categories, no training needed.")

with st.expander("How this works"):
    st.markdown(
        """
        This uses **transfer learning**: instead of training a model from
        scratch (which needs millions of labeled images), we reuse
        **MobileNetV2**, already trained by Google on the ImageNet dataset
        (1.2 million images, 1,000 categories).

        1. Your image is resized to 224×224 pixels — the fixed size this
           network expects.
        2. It's converted into a batch of numbers and scaled the same way
           MobileNetV2 was trained (`preprocess_input`).
        3. The network outputs a probability for each of its 1,000 known
           categories — we show the top 5.
        """
    )

st.subheader("Provide an image")
uploaded_file = st.file_uploader("Upload a photo (jpg/png)", type=["jpg", "jpeg", "png"])

use_sample = False
if uploaded_file is None:
    use_sample = st.checkbox("No photo handy — use a sample image instead", value=False)

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif use_sample:
    if not os.path.exists(FALLBACK_IMAGE_PATH):
        urllib.request.urlretrieve(FALLBACK_IMAGE_URL, FALLBACK_IMAGE_PATH)
    image = Image.open(FALLBACK_IMAGE_PATH).convert("RGB")

if image is not None:
    resized_image = image.resize((224, 224))
    st.image(resized_image, caption="Input image (resized to 224x224)")

    if st.button("Analyze Image", type="primary"):
        with st.spinner("Running the network..."):
            top_5, elapsed_ms = predict_image(model, resized_image)

        st.success(f"Prediction took {elapsed_ms:.1f} ms")
        st.subheader("Top 5 predictions")
        df = pd.DataFrame(top_5, columns=["Label", "Probability"])
        df["Probability"] = (df["Probability"] * 100).round(2).astype(str) + "%"
        st.table(df)

        best_label, best_prob = top_5[0]
        st.metric("Best guess", best_label.replace("_", " ").title(), f"{best_prob * 100:.1f}% confident")
else:
    st.info("Upload a photo above, or check the box to try a sample image.")

st.caption("Part of the AI Playground: 4 Real-World AI Projects series.")
