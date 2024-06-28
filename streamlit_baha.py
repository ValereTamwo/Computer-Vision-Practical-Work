import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage import exposure
from skimage.measure import moments_hu  # Correct import from skimage.measure
from sklearn.decomposition import PCA

# Charger les données MNIST depuis OpenML
mnist = fetch_openml('mnist_784', version=1)

# Accéder aux données et aux cibles
X, y = mnist.data, mnist.target.astype(int)

# Convertir les données en tableau numpy et les redimensionner
X = np.array(X).reshape((-1, 28, 28))

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fonction pour extraire les moments de Hu à partir d'une image
def hu_moments_features(X):
    features = []
    for image in X:
        moments = moments_hu(image)  # Extraire les moments de Hu de l'image
        features.append(moments)
    return np.array(features)

# Fonction pour extraire les HOG (Histogrammes de gradient orientés) à partir d'une image
def hog_features(X):
    features = []
    for image in X:
        hog_features = hog(image, block_norm='L2-Hys', pixels_per_cell=(8, 8))
        features.append(hog_features)
    return np.array(features)

# Extraire les caractéristiques avec Moments de Hu pour les ensembles d'entraînement et de test
print("Extraction des caractéristiques avec Moments de Hu...")
X_train_hu = hu_moments_features(X_train)
X_test_hu = hu_moments_features(X_test)

# Entraînement et évaluation du SVM avec Moments de Hu
svm_hu = SVC(kernel='linear', random_state=42)
svm_hu.fit(X_train_hu, y_train)
y_pred_hu = svm_hu.predict(X_test_hu)
accuracy_hu = accuracy_score(y_test, y_pred_hu)
print(f"Précision du SVM avec Moments de Hu : {accuracy_hu:.4f}")

# Extraire les caractéristiques avec HOG pour les ensembles d'entraînement et de test
print("Extraction des caractéristiques avec HOG...")
X_train_hog = hog_features(X_train)
X_test_hog = hog_features(X_test)

# Entraînement et évaluation du SVM avec HOG
svm_hog = SVC(kernel='linear', random_state=42)
svm_hog.fit(X_train_hog, y_train)
y_pred_hog = svm_hog.predict(X_test_hog)
accuracy_hog = accuracy_score(y_test, y_pred_hog)
print(f"Précision du SVM avec HOG : {accuracy_hog:.4f}")

# Comparaison des précisions dans un tableau
print("\nComparaison des précisions :")
print("-" * 50)
print("Méthode\t\t\t\tPrécision")
print("-" * 50)
print(f"Moments de Hu\t\t\t{accuracy_hu:.4f}")
print(f"HOG\t\t\t\t{accuracy_hog: