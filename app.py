import os


from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# ---------- settings ----------
MODEL_PATH = Path(__file__).parent / "scene_cnn_model_small.keras"
IMAGE_SIZE = (150, 150)
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

st.set_page_config(page_title="Scene Classifier", page_icon="🏞️")


@st.cache_resource(show_spinner="Loading the model...")
def load_scene_model():
    return load_model(MODEL_PATH,compile = False)


model = load_scene_model()

# ---------- sidebar ----------
with st.sidebar:
    st.header("About this project")
    st.write("A CNN I trained from scratch to recognise six kinds of scenes.")
    st.metric("Test accuracy", "80.3%")
    st.caption("Measured on 456 photos the model never saw during training.")

    st.subheader("Scenes it knows")
    for name in CLASS_NAMES:
        st.write(f"- {name.capitalize()}")

    st.caption("It can mix up sea, glacier and mountain, and street and buildings.")

# ---------- main page ----------
st.title("🏞️ Scene Image Classifier")
st.write("Upload a photo and the model will tell you which scene it thinks it is.")

uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # open the photo, fix rotation, and make sure it has 3 colour channels
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img).convert("RGB")

    # same steps as in training: 150x150, values between 0 and 1, batch of one
    arr = np.array(img.resize(IMAGE_SIZE), dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    best = int(np.argmax(probs))

    left, right = st.columns(2)

    with left:
        st.image(img, caption="Your photo")

    with right:
        st.subheader("Prediction")
        st.metric("Scene", CLASS_NAMES[best].capitalize())
        st.write(f"Confidence: {probs[best]:.1%}")

        if probs[best] < 0.60:
            st.warning("I'm not very sure about this one.")

        st.write("**All scores**")
        for name, p in zip(CLASS_NAMES, probs):
            st.progress(float(p), text=f"{name.capitalize()}: {p:.1%}")
