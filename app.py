import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Set Streamlit page config
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

# Load the trained model
@st.cache_resource
def load_cnn_model():
    return load_model("my_cat_dog_classifier.h5")

model = load_cnn_model()

# Title
st.title("🐶🐱 Cat vs Dog Image Classifier")

# Image uploader
uploaded_file = st.file_uploader("Upload an image of a cat or dog...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0  # Normalize
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]  # Remove alpha channel if present
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction >= 0.5 else "Cat 🐱"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    # Show result
    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")
