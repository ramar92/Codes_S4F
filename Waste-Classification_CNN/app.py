"""
Streamlit app to deploy the Organic vs Recyclable waste-classification CNN.

Run with:
    streamlit run app.py

The app expects a trained Keras model file named `waste_classifier_model.h5`
in the same folder (produced by the accompanying notebook,
Organic_Recyclable_CNN_Classification.ipynb). If the model file is missing,
the app will still load and tell you how to generate it.
"""

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = "waste_classifier_model.h5"
IMG_SIZE = (224, 224)
# Matches the class order produced by ImageDataGenerator.flow_from_directory
# on a dataset with TRAIN/O (Organic) and TRAIN/R (Recyclable) folders,
# since Keras sorts class folders alphabetically -> {'O': 0, 'R': 1}.
IDX_TO_LABEL = {0: "Organic", 1: "Recyclable"}

st.set_page_config(page_title="Organic vs Recyclable Classifier", page_icon="♻️", layout="centered")

st.title("♻️ Organic vs Recyclable Waste Classifier")
st.write(
    "Upload an image of a waste item and the CNN model will predict whether "
    "it's **Organic** or **Recyclable**."
)


@st.cache_resource
def load_trained_model(path: str):
    if not os.path.exists(path):
        return None
    from tensorflow.keras.models import load_model
    return load_model(path)


model = load_trained_model(MODEL_PATH)

if model is None:
    st.warning(
        f"Couldn't find `{MODEL_PATH}` in this folder. Train the model using "
        "`Organic_Recyclable_CNN_Classification.ipynb` first (it saves the file "
        "automatically at the end of training), then place it next to `app.py` "
        "and refresh this page."
    )

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if model is not None:
        with st.spinner("Classifying..."):
            img_array = np.array(image)
            resized = cv2.resize(img_array, IMG_SIZE)
            batch = np.reshape(resized, [-1, IMG_SIZE[0], IMG_SIZE[1], 3]) / 255.0

            predictions = model.predict(batch)
            class_idx = int(np.argmax(predictions))
            confidence = float(np.max(predictions)) * 100
            label = IDX_TO_LABEL.get(class_idx, str(class_idx))

        if label == "Organic":
            st.success(f"🟢 Prediction: **{label}** ({confidence:.1f}% confidence)")
        else:
            st.info(f"🔵 Prediction: **{label}** ({confidence:.1f}% confidence)")

        with st.expander("Raw prediction scores"):
            st.write(predictions.tolist())
    else:
        st.stop()

st.markdown("---")
st.caption(
    "Model: custom CNN (3 conv blocks + dense head), trained on 224x224 RGB images "
    "of Organic and Recyclable waste."
)
