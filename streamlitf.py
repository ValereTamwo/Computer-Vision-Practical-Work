import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib



# Streamlit app
st.title('Handwritten Digit Recognition')

uploaded_file = st.file_uploader("Upload an image of a digit (28x28 pixels)", type="png")

if uploaded_file is not None:
    # Load the image
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)

