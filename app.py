import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================
# Dataset Sederhana
# =====================

data = {
    "text": [
        "produk ini sangat bagus",
        "saya sangat puas",
        "pelayanan memuaskan",
        "barang berkualitas tinggi",
        "pengiriman cepat",

        "produk ini buruk",
        "saya kecewa",
        "pelayanan sangat jelek",
        "barang rusak",
        "pengiriman lambat",

        "luar biasa dan keren",
        "sangat membantu",
        "rekomendasi terbaik",

        "tidak sesuai harapan",
        "sangat mengecewakan",
        "tidak akan beli lagi"
    ],
    "label": [
        "Positif",
        "Positif",
        "Positif",
        "Positif",
        "Positif",

        "Negatif",
        "Negatif",
        "Negatif",
        "Negatif",
        "Negatif",

        "Positif",
        "Positif",
        "Positif",

        "Negatif",
        "Negatif",
        "Negatif"
    ]
}

df = pd.DataFrame(data)

# =====================
# Train Model
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# =====================
# Streamlit UI
# =====================

st.set_page_config(
    page_title="Analisis Sentimen Sederhana",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Analisis Sentimen NLP Tradisional")

st.write("""
Aplikasi ini menggunakan:
- TF-IDF Vectorizer
- Naive Bayes Classifier
- Scikit-Learn
""")

st.metric("Akurasi Testing", f"{accuracy:.2%}")

st.subheader("Dataset")

st.dataframe(df)

st.subheader("Prediksi Sentimen")

user_input = st.text_area(
    "Masukkan kalimat:",
    placeholder="Contoh: produk ini sangat bagus"
)

if st.button("Prediksi"):

    if user_input.strip():

        prediction = model.predict([user_input])[0]

        probability = model.predict_proba([user_input])[0]

        classes = model.classes_

        st.success(f"Sentimen: **{prediction}**")

        prob_df = pd.DataFrame({
            "Kelas": classes,
            "Probabilitas": probability
        })

        st.dataframe(prob_df)

    else:
        st.warning("Masukkan teks terlebih dahulu.")