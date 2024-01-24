import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Charger le dataset
data = pd.read_csv('dataset.csv')

# Charger le modèle KNN
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Comment'])
X_train, X_test, Y_train, Y_test = train_test_split(X, data['mood'], test_size=0.2, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, Y_train)


# Afficher les données dans la page principale
st.title("Classification des Commentaires Positifs ou Negatifs avec KNN")

# Utiliser un formulaire au lieu d'un champ de texte
with st.form("commentaire_form"):
    # Afficher le formulaire d'entrée de texte
    user_input_main = st.text_input("Entrez votre Commentaire ici")

    # Bouton de prédiction centré avec une couleur simple
    bouton_prediction = st.form_submit_button(":blue[Prédire le Type de Commentaire]", help="Appuyez pour prédire")

# Conditions pour la prédiction et le stockage des commentaires
if bouton_prediction:
    if user_input_main:
        vectorized_text_main = vectorizer.transform([user_input_main])
        prediction_main = knn_model.predict(vectorized_text_main)
        if prediction_main[0] == "Positif":
            st.success(f"Commentaire {prediction_main[0]} !!!", icon="✅")
        else:
            st.warning(f"Commentaire {prediction_main[0]} !!!", icon="❌")
    else:
        st.warning("Veuillez entrer un commentaire pour la prédiction.")


