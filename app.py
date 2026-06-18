import streamlit as st
import pandas as pd
import re
import random
import torch
from transformers import pipeline

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Gibran Sentiment Analyzer",
    page_icon="🤖",
    layout="centered"
)

# Title & Deskripsi Aplikasi
st.title("🤖 Gibran Sentiment Analyzer App")
st.markdown("""
Aplikasi web ini digunakan untuk menguji sentimen ulasan publik secara interaktif 
menggunakan pendekatan **NLP Tradisional** dan **NLP Modern (Transformer)**.
Dataset dilatih menggunakan data komentar YouTube Gibran Rakabuming Raka.
""")

# ==========================================
# 2. IMPLEMENTASI CACHING MODEL (OPTIMASI PERFORMA)
# ==========================================
@st.cache_resource
def load_modern_transformer_model():
    """Memuat model IndoBERT Sentiment secara tersimpan di cache RAM agar tidak reload ulang."""
    # Menggunakan model pre-trained IndoBERT dari wiraa/indobert-sentiment (~500MB)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="wiraa/indobert-sentiment",
        tokenizer="wiraa/indobert-sentiment",
        device=-1  # Menggunakan CPU mode agar aman di cloud gratis
    )
    return sentiment_pipeline

# Memanggil fungsi load model di background saat aplikasi dibuka
with st.spinner("Sedang memuat Model Transformer IndoBERT ke memori... Mohon tunggu sebentar."):
    modern_pipeline = load_modern_transformer_model()

# ==========================================
# 3. FUNGSI UTAMA TEXT PREPROCESSING (Materi Week 9 & 10)
# ==========================================
# Kamus Normalisasi Slang Bahasa Tradisional Indonesia
slang_dict = {
    "yg": "yang", "ga": "tidak", "gak": "tidak", "gk": "tidak",
    "gw": "saya", "gue": "saya", "lu": "kamu", "lo": "kamu",
    "kalo": "kalau", "dgn": "dengan", "bgt": "banget", "bener": "benar",
    "pake": "pakai", "pke": "pakai", "sampe": "sampai", "aja": "saja",
    "sdh": "sudah", "dah": "sudah", "udh": "sudah", "karna": "karena",
    "krn": "karena", "tp": "tetapi", "tapi": "tetapi", "moga": "semoga",
    "jd": "jadi", "jdi": "jadi", "wapres": "wakil presiden"
}

def preprocess_traditional_pipeline(text):
    """Pembersihan Teks Intensif untuk Model Tradisional Statistik."""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)          # Hapus Mention
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # Hapus URL
    text = re.sub(r'[^a-zA-Z\s]', '', text)             # Hapus Angka dan Tanda Baca
    tokens = text.split()
    normalized_tokens = [slang_dict.get(word, word) for word in tokens]
    return " ".join(normalized_tokens)

def preprocess_modern_pipeline(text):
    """Pembersihan Ringan (Light Preprocessing) Menjaga Konteks Kalimat untuk BERT."""
    if not isinstance(text, str): 
        return ""
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s\?\!\\.,]', '', text)         # Tetap simpan tanda seru & tanya untuk penekanan emosi sentimen
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 4. IMPLEMENTASI LOGIK KELAS SIMULASI TRADISIONAL LEXICON
# ==========================================
positive_words = {
    "hebat", "bagus", "keren", "salut", "setuju", "maju", "cerdas", 
    "pintar", "top", "mantap", "dukung", "baik", "semangat", "sukses"
}
negative_words = {
    "jelek", "buruk", "kecewa", "gagal", "bodoh", "lemah", "kosong", 
    "ironis", "sindir", "malas", "benci", "marah", "bohong", "curang"
}

def predict_traditional_lexicon(text):
    cleaned = preprocess_traditional_pipeline(text)
    pos_count = sum(1 for word in positive_words if word in cleaned)
    neg_count = sum(1 for word in negative_words if word in cleaned)
    
    if pos_count > neg_count: 
        return "POSITIF", 0.85
    elif neg_count > pos_count: 
        return "NEGATIF", 0.85
    else: 
        return "NETRAL", 0.50

# ==========================================
# 5. PEMBUATAN NAVIGASI ANTARMUKA MENGGUNAKAN TABS
# ==========================================
tab1, tab2 = st.tabs(["⚡ Model Tradisional (Lexicon Based)", "🔥 Model Modern (IndoBERT Transformer)"])

# ----- IMPLEMENTASI TAB 1: TRADISIONAL -----
with tab1:
    st.header("Analisis Sentimen Tradisional")
    st.caption("Memanfaatkan normalisasi teks bahasa gaul dan pencocokan bobot kata kunci.")
    
    input_teks_trad = st.text_area(
        "Masukkan ulasan politik (Contoh: Gibran penampilannya keren bgt bener-bener mantap!):", 
        key="txt_trad"
    )
    
    if st.button("Jalankan Model Tradisional", type="primary"):
        if input_teks_trad.strip() == "":
            st.warning("Mohon masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Menghitung teks..."):
                hasil_label, akurasi = predict_traditional_lexicon(input_teks_trad)
                teks_terproses = preprocess_traditional_pipeline(input_teks_trad)
                
                st.subheader("Hasil Analisis:")
                st.write(f"**Teks Hasil Preprocessing:** `{teks_terproses}`")
                
                if hasil_label == "POSITIF":
                    st.success(f"Sentimen Terdeteksi: **{hasil_label}** (Skor Pendukung: {akurasi})")
                elif hasil_label == "NEGATIF":
                    st.error(f"Sentimen Terdeteksi: **{hasil_label}** (Skor Pendukung: {akurasi})")
                else:
                    st.info(f"Sentimen Terdeteksi: **{hasil_label}** (Skor Pendukung: {akurasi})")

# ----- IMPLEMENTASI TAB 2: MODERN TRANSFORMER -----
with tab2:
    st.header("Analisis Sentimen Modern")
    st.caption("Memanfaatkan model Deep Learning IndoBERT yang memahami makna kontekstual struktur kalimat utuh.")
    
    input_teks_mod = st.text_area(
        "Masukkan ulasan politik (Contoh: Gak nyangka performa wapres gibran luar biasa!):", 
        key="txt_mod"
    )
    
    if st.button("Jalankan Model Modern BERT", type="primary"):
        if input_teks_mod.strip() == "":
            st.warning("Mohon masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Model IndoBERT sedang menganalisis struktur kalimat..."):
                # 1. Jalankan pembersihan teks ala Transformer
                teks_terproses_mod = preprocess_modern_pipeline(input_teks_mod)
                
                # 2. Kirim ke pipeline model transformer
                prediksi_bert = modern_pipeline(teks_terproses_mod)
                label_bert = prediksi_bert[0]['label'].upper()
                skor_bert = prediksi_bert[0]['score']
                
                st.subheader("Hasil Analisis Deep Learning:")
                st.write(f"**Teks Hasil Preprocessing:** `{teks_terproses_mod}`")
                
                # Visualisasi Kondisional Berdasarkan Output Label Model
                if "POS" in label_bert:
                    st.success(f"Sentimen Terdeteksi: **POSITIF** (Confidence Score: {skor_bert:.4f})")
                    st.balloons()
                elif "NEG" in label_bert:
                    st.error(f"Sentimen Terdeteksi: **NEGATIF** (Confidence Score: {skor_bert:.4f})")
                else:
                    st.info(f"Sentimen Terdeteksi: **NETRAL** (Confidence Score: {skor_bert:.4f})")