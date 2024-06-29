import numpy as np
import streamlit as st
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.measure import moments_hu
from PIL import Image
import io

# Fonction pour extraire les moments de Hu à partir d'une image
def hu_moments_features(image):
    gray_image = rgb2gray(image)
    moments = moments_hu(gray_image)  # Extraire les moments de Hu de l'image
    return moments

# Fonction pour extraire les HOG (Histogrammes de gradient orientés) à partir d'une image
def hog_features(image):
    gray_image = rgb2gray(image)
    features = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(8, 8))
    return features

# Fonction principale de l'application Streamlit
def main():
    st.title("Extraction de Caractéristiques d'Images")

    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)

        st.image(image, caption='Image téléchargée.', use_column_width=True)
        st.write("")
        st.write("Extraction des caractéristiques...")

        # Extraction des caractéristiques avec Moments de Hu
        hu_features = hu_moments_features(image)
        st.write("Moments de Hu:")
        st.write(hu_features)

        # Extraction des caractéristiques avec HOG
        hog_features_extracted = hog_features(image)
        st.write("HOG Features:")
        st.write(hog_features_extracted)

if __name__ == "__main__":
    main()
