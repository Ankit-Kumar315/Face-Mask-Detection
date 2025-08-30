import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io

# Load model once & cache it
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Ankit_Kumar_model.h5")

model = load_model()

# UI
st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="centered")
st.title("😷 Face Mask Detection App")
st.write("Upload an image or take a photo, and the app will predict whether the person is wearing a mask.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Camera input
camera_file = st.camera_input("Or take a picture with your camera")

# Prediction logic
def preprocess_image(image: Image.Image):
    img_array = np.array(image)

    # If grayscale, convert to RGB
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Resize to model input size
    img_resized = cv2.resize(img_array, (224, 224))
    img_resized = img_resized / 255.0  # normalize
    img_input = np.expand_dims(img_resized, axis=0)  # shape (1, 224, 224, 3)

    return img_input

# Choose input source (file upload OR camera)
image_file = uploaded_file or camera_file

if image_file is not None:
    if isinstance(image_file, bytes):  # camera_input returns bytes
        image = Image.open(io.BytesIO(image_file))
    else:  # file_uploader returns UploadedFile
        image = Image.open(image_file)

    st.image(image, caption="Selected Image", use_container_width=True)

    img_input = preprocess_image(image)

    # Make prediction
    prediction = model.predict(img_input)
    prob = prediction[0][0]

    label = "Mask 😷" if prob < 0.5 else "No Mask ❌"
    confidence = (1 - prob) if prob < 0.5 else prob

    st.subheader(f"Prediction: **{label}**")
    st.progress(int(confidence * 100))
    st.write(f"Confidence: **{confidence:.2%}**")
