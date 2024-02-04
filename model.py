import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


import string
def clean_sentences(text):
  # Conversion en minuscules
    text = str(text).lower()

    # Suppression de la ponctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    if text:
      return text
    else:
      print("Un probleme")

# Test du modèle
def predict(text):
    clean_sentences(text)

    vectorised_text = vectorizer.transform(text)

    prediction = knn_model.predict(vectorised_text)

    return prediction[0]


# Charger le dataset
data = pd.read_csv('dataset.csv')

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Comment'])

# Modele
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X, data['mood'])


# Afficher les données dans la page principale
st.title("Classification des Commentaires Positifs ou Negatifs avec KNN")

# Utiliser un formulaire au lieu d'un champ de texte
with st.form("commentaire_form"):
    # Afficher le formulaire d'entrée de texte
    inout = st.text_input("Entrez votre Commentaire ici")

    # Bouton de prédiction centré avec une couleur simple
    bouton_prediction = st.form_submit_button("Prédire le Type de Commentaire", help="Appuyez pour prédire")

# Conditions pour la prédiction et le stockage des commentaires
if bouton_prediction:
    if input:
        result = predict(input)
        if result == "Positif":
            st.success(f"Commentaire {result} !!!", icon="✅")
        else:
            st.warning(f"Commentaire {result} !!!", icon="❌")
    else:
        st.warning("Veuillez entrer un commentaire pour la prédiction.")


