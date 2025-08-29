# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "garbage_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class mapping (make sure this matches your training order)
class_names = ["plastic", "organic", "metal"]

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((128, 128))           # same size as training
    img_array = np.array(img) / 255.0        # normalize
    img_array = np.expand_dims(img_array, 0) # add batch dimension
    return img_array

# ----------------------------
# Streamlit frontend
# ----------------------------
st.title("♻️ Garbage Classification App")
st.write("Upload an image of garbage (plastic, organic, or metal), and the model will classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show results
    st.subheader("Prediction")
    st.write(f"**Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
