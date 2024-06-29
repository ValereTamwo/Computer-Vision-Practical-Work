import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import random
from skimage.feature import hog
from PIL import Image


import joblib

# Fonction pour afficher les images avec points clés
def plot_keypoints(images, keypoints_list, num_images=4):
    plt.figure(figsize=(5, 5))
    for i in range(num_images):
        img = images[i].reshape(28, 28)
        keypoints = keypoints_list[i]
        img_with_keypoints = cv2.drawKeypoints(np.uint8(img), keypoints, None)
        plt.subplot(2, 2, i+1)
        plt.imshow(img_with_keypoints, cmap='gray')
        plt.axis('off')
    plt.show()
    st.pyplot(plt)
    
def harris_corner_detector(image, block_size=2, ksize=3, k=0.04):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    return image

# Fonction pour extraire les caractéristiques SIFT
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def extract_sift_feature(image):
    # Convertir l'image en CV_8U si elle n'est pas déjà en ce format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def predict_new_image(image , clf):
    _, descriptors = extract_sift_feature(image)
    if descriptors is None:
        descriptors = np.zeros((1, 128))
    flat = descriptors.flatten()
    padded =np.pad(flat, (0, 3200 - len(flat)), 'constant') 
    return clf.predict([padded])


def hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

def hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(gray, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True,
                              block_norm='L2-Hys')
    return features, hog_image


def feature_matching(img1, img2):
    
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    img1_gray = cv2.cvtColor(np.array(img1_array), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(np.array(img2_array), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Appliquer le ratio test de David Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Dessiner les matches
    img_matches = cv2.drawMatches(img1_array, keypoints1, img2_array, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

# Interface Streamlit
(X_train, y_train), (X_test, y_test) = mnist.load_data()

st.title("Reconnaissance d'Écriture Manuscrite")

st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Sélectionnez une section", 
                              ["Accueil", "Extraction de Caractéristiques Manuelles", "Modèles de Machine Learning", "Modèles de Deep Learning", "Predictions et Demo", "Conclusion et Perspectives"])

if option == "Accueil":
    st.header("Présentation du Travail sur la Reconnaissance d'Écriture Manuscrite")
    st.write("""
    Bienvenue sur notre application de reconnaissance d'écriture manuscrite. Ce projet explore différentes méthodes pour la reconnaissance de caractères manuscrits en utilisant des datasets variés.
    """)

    st.header("Description des Datasets Utilisés")
    st.image('rime.ppm', caption='Exemple d\'image du dataset Rime et IAM', use_column_width=True)
    st.write("""
    - **MNIST** : Le dataset MNIST est constitué de 70 000 images de chiffres manuscrits (0-9) de 28x28 pixels. Il est largement utilisé pour l'entraînement et l'évaluation des modèles de reconnaissance d'écriture manuscrite.
    
    - **Rime** : Le dataset Rime contient des échantillons de texte manuscrit pour la reconnaissance de caractères dans divers scripts et langues.
    
    - **IAM** : Le dataset IAM contient des lignes de texte manuscrit anglais et est utilisé pour entraîner et évaluer les modèles de reconnaissance de texte manuscrit.
    """)

    st.header("Contributions des Membres du Groupe")
    st.write("""
    - **Utilisateur** : J'ai travaillé sur l'application des techniques de machine learning classiques en utilisant le dataset MNIST. Mon objectif était d'extraire manuellement des caractéristiques et de les utiliser pour la classification des chiffres manuscrits.
    
    - **Brel** : Brel a entraîné un modèle de deep learning et a créé une interface web avec Streamlit pour afficher les étapes de travail et réaliser des prédictions en utilisant le dataset Rime.
    
    - **Bahouaddyn** : Bahouaddyn s'est concentré sur l'extraction manuelle des caractéristiques en utilisant plusieurs méthodes et a comparé la pertinence de ces caractéristiques pour la reconnaissance d'écriture manuscrite.
    
    - **Fideline Uy1** : Fideline a effectué un travail similaire à celui de Brel mais en utilisant le dataset IAM. Elle a également visualisé les caractéristiques extraites par le modèle de deep learning.
    """)

elif option == "Extraction de Caractéristiques Manuelles":
    st.header("Extraction de Caractéristiques Manuelles")
    st.write("Présentation des différentes méthodes d'extraction de caractéristiques manuelles.")
    st.image('HOC.png')
    
    st.subheader('Extraction en temp Reel')
    
    method = st.selectbox("Méthode", ["SIFT", "Harris Corner Detector", "Moments de Hu", "HOG"])

    uploaded_imag = st.file_uploader("Téléchargez une image...", type=["png", "jpg", "jpeg"])

    if uploaded_imag is not None:
        file_bytes = np.asarray(bytearray(uploaded_imag.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        if method == "SIFT":
            keypoints, _ = extract_sift_features(img)
            img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)
            st.image(img_with_keypoints, caption='Image avec points clés SIFT', use_column_width=True)
        elif method == "Harris Corner Detector":
            img_with_corners = harris_corner_detector(img)
            st.image(img_with_corners, caption='Image avec Harris Corners', use_column_width=True)
        elif method == "Moments de Hu":
            hu_features = hu_moments(img)
            st.write("Moments de Hu:", hu_features)
        elif method == "HOG":
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hog_features, hog_image = hog(gray, block_norm='L2-Hys', pixels_per_cell=(8, 8), visualize=True)
            # Normalize hog_image
            hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())
        
            st.image(hog_image, caption='Image avec HOG', use_column_width=True)
            st.write("Caractéristiques HOG : ", hog_features)
            
            
    st.subheader("Sélectionnez deux images du dataset MNIST pour le feature matching :")
    uploaded_img1 = st.file_uploader("Téléchargez une image 1...", type=["png", "jpg", "jpeg"])
    if uploaded_img1 is not None:
        img1 = Image.open(uploaded_img1)
        st.image(img1, caption="image 1", use_column_width=True)

    uploaded_img2 = st.file_uploader("Téléchargez une image 2...", type=["png", "jpg", "jpeg"])
    if uploaded_img2 is not None:
        img2 = Image.open(uploaded_img2)
        st.image(img2, caption="image 2", use_column_width=True)
    
    # random_indices = random.sample(range(len(X_test)), 10)
    # idx1 = st.selectbox("Index de la première image :", random_indices)
    # idx2 = st.selectbox("Index de la deuxième image :", random_indices)

    if st.button("Faire le Feature Matching"):
        img1 = Image.open(uploaded_img1)
        img2 = Image.open(uploaded_img2)
        matched_img = feature_matching(img1, img2)
        
        # img11 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        # img22 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        # matched_img = feature_matching(img1, img2)
        st.image(matched_img, caption='Feature Matching entre deux images', use_column_width=True)

elif option == "Modèles de Machine Learning":
    st.header("Modèles de Machine Learning Classiques avec Feature Extraction")
    st.write("Présentation des modèles de machine learning utilisés pour la reconnaissance d'écriture manuscrite.")
    st.write("Extraction des features avec SIFT : cas de test MNIST")
    
    if st.button("Visualiser les caractéristiques"):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = X_train[:4]
        keypoints_list = []
        descriptors_list =[]
        
        for img in X:
            keypoints, descriptors = extract_sift_features(img)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
        plot_keypoints(X, keypoints_list)   
        
        for i in range(4):
            st.write(f"Image {i+1}:")
            st.write("Keypoints:", keypoints_list[i])
            st.write("Descriptors shape:", descriptors_list[i].shape)
            st.write(descriptors_list[i])

    st.write("Upload your own handwritten image for feature extraction:")
    uploaded_image = st.file_uploader("Choose an image...", type="png")
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, _ = extract_sift_features(gray)
        img_with_keypoints = cv2.drawKeypoints(gray, keypoints, None)
        st.image(img_with_keypoints, caption='Uploaded Image with SIFT Keypoints', use_column_width=True)
        
    st.write("### Utilisation des descripteurs extraits pour la prédiction")
    st.write("""
    Une fois les descripteurs extraits des images, nous les transformons pour les utiliser avec un modèle de machine learning classique comme SVM. Voici les étapes clés :

    - **Aplatir et pad les descripteurs** : Les descripteurs extraits peuvent avoir des tailles différentes pour chaque image. Pour les utiliser avec un SVM, nous devons les aplatir et les padder afin d'obtenir une matrice de descripteurs de taille fixe.
    - **Entraîner le modèle SVM** : Nous utilisons les descripteurs extraits et transformés pour entraîner un modèle SVM. Les étiquettes des images (par exemple, les chiffres 0-9 pour le dataset MNIST) sont utilisées comme classes cibles.
    - **Prédiction** : Pour faire des prédictions sur de nouvelles images, nous extrayons d'abord les descripteurs, les transformons de la même manière (aplatir et pad), puis utilisons le modèle SVM entraîné pour prédire les classes.

    Voici un exemple de code pour entraîner un SVM avec les descripteurs extraits :
    """)

    st.code("""
    import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Prétraitement des images: Normalisation et conversion en type float32
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Fonction pour extraire les descripteurs SIFT
def extract_sift_features(image):
    # Convertir l'image en CV_8U si elle n'est pas déjà en ce format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def transform_descriptors(descriptors_list):
    # Aplatir chaque descripteur en un seul vecteur
    flattened_descriptors = [d.flatten() if d is not None else np.array([]) for d in descriptors_list]
    max_length = max(len(d) for d in flattened_descriptors)
    padded_descriptors = [np.pad(d, (0, max_length - len(d)), 'constant') for d in flattened_descriptors]
    return np.array(padded_descriptors)

# Extraire les descripteurs pour l'ensemble d'entraînement
X_train_descriptors = [extract_sift_features(img)[1] for img in X_train]
X_test_descriptors = [extract_sift_features(img)[1] for img in X_test]

# Entraîner le modèle SVM
clf = make_pipeline(StandardScaler(), LogisticRegression())
clf.fit(X_train_transformed, y_train)

    """)

    st.write("""
    Ce processus permet de tirer parti des caractéristiques extraites manuellement pour améliorer la précision de la reconnaissance d'écriture manuscrite en utilisant des techniques de machine learning classiques.
    """)

elif option == "Modèles de Deep Learning":
    st.header("Modèles de Deep Learning")
    st.write("Présentation des modèles de deep learning utilisés pour la reconnaissance d'écriture manuscrite.")
    # Ajoutez ici le code pour entraîner et évaluer les modèles de deep learning
    # Exemple de visualisation des caractéristiques extraites par un modèle
    if st.button("Visualiser les caractéristiques"):
        # Code pour visualiser les caractéristiques extraites par le modèle de deep learning
        st.write('Visualisation des caractéristiques extraites par le modèle de deep learning.')

elif option == "Predictions et Demo":
    st.header("Prédictions et Démonstrations")

    # Partie Machine Learning Classique
    st.subheader("Prédictions avec Modèles de Machine Learning Classiques")
    
    # Affichage des classes de MNIST
    st.write("Classes du dataset MNIST :")
    classes = list(range(10))  # Les classes vont de 0 à 9 pour MNIST
    st.write('Matrice de Performance')
    st.image('matrix.png', caption='Matrice de Confusions', use_column_width=True)
    st.write(classes)
    
    st.write("Téléchargez votre propre image manuscrite pour la prédiction :")
    uploaded_image = st.file_uploader("Choisissez une image...", type="png")
    
    if uploaded_image is not None:
        
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Redimensionner l'image à 28x28
            resized_img = cv2.resize(gray, (28, 28))
            
            keypoints, descriptors = extract_sift_features(gray)
            img_with_keypoints = cv2.drawKeypoints(gray, keypoints, None)
            st.image(img_with_keypoints, caption='Image téléchargée avec les points clés SIFT', use_column_width=True)
            st.write("Keypoints:", len(keypoints))
            st.write("Descriptors shape:", descriptors.shape)
            st.write(descriptors)

            # Transformer les descripteurs pour la prédiction
            clf = joblib.load('./model/logres_digit.pkl')
            # Prédiction
            pred = predict_new_image(resized_img, clf)
            st.write(f"Prédiction : {pred}")
            
                    # Sélection d'une image du dataset MNIST
    st.write("Sélectionnez une image du dataset MNIST pour la prédiction :")
    clf = joblib.load('./model/logres_digit.pkl')
    
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    random_indices = random.sample(range(len(X_test)), 10)
    selected_image_index = st.selectbox("Sélectionnez l'index de l'image :", random_indices)
    
    selected_image = X_test[selected_image_index]
    st.image(selected_image, caption=f'Image MNIST index {selected_image_index}', use_column_width=True)
    
    if st.button("Prédire l'image sélectionnée"):                
        pred = predict_new_image(selected_image, clf)
        st.write(f"Prédiction pour l'image sélectionnée : {pred[0]}")
        
elif option == "Conclusion et Perspectives":
    st.header("Conclusion et Perspectives")
    st.write("Résumé des résultats obtenus, discussion sur les défis rencontrés et les futures améliorations possibles.")
