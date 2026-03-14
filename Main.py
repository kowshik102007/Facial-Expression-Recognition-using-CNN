import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
# Load model
@st.cache_resource
def load_fer_model():
    return load_model(r"C:\acm\fer_model.h5")

model = load_fer_model()
labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

st.title("😊 Facial Expression Recognition")

file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if file:
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48,48)) / 255.0
    face = face.reshape(1, 48, 48, 1)

    preds = model.predict(face)
    idx = np.argmax(preds)

    st.subheader(f"Prediction: {labels[idx]} 😀")
    st.write("Confidence:", float(np.max(preds)))
