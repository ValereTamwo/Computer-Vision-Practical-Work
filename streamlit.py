import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib

# Load the trained model
model = joblib.load('model/digit_recognizer_model_svm.pkl')


# Function to extract SIFT features from input image
def extract_sift_features_from_image(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        max_descriptors = 128  # assuming max 128 descriptors for simplicity
        descriptors_padded = np.pad(descriptors, ((0, max_descriptors - len(descriptors)), (0, 0)), 'constant')
        return descriptors_padded.flatten().reshape(1, -1)
    else:
        return np.zeros((1, sift.descriptorSize() * 128))  # assuming max 128 descriptors


# Streamlit app
st.title('Handwritten Digit Recognition')

uploaded_file = st.file_uploader("Upload an image of a digit (28x28 pixels)", type="png")

if uploaded_file is not None:
    # Load the image
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)

    # Extract SIFT features
    features = extract_sift_features_from_image(img)

    # Predict the digit
    prediction = model.predict(features)

    st.write(f'Predicted Digit: {prediction[0]}')
