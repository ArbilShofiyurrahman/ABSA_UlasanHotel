import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Menghapus angka dan karakter non-huruf
    return text

# Fungsi Normalisasi (Contoh)
def normalize_negation(text):
    negation_patterns = {
        r'\btidak bersih\b': 'kotor',
        r'\btidak teratur\b': 'berantakan',
        # ... (tambahkan pola lainnya)
    }
    for pattern, replacement in negation_patterns.items():
        text = re.sub(pattern, replacement, text)
    return text

# Fungsi Preprocessing
def preprocess_text(text, stopword_model, stemmer_model):
    text = text.lower()  # Casefolding
    text = clean_text(text)
    text = normalize_negation(text)  # Normalisasi
    text = stopword_model.remove(text)  # Stopword Removal
    text = stemmer_model.stem(text)  # Stemming
    return text

# Memuat Model
try:
    stopword_model = joblib.load('stopword_remover_model.pkl')
    stemmer_model = joblib.load('stemmer_model.pkl')
    tfidf_aspek = joblib.load('tfidf_aspek.pkl')
    tfidf_sentimen = joblib.load('tfidf_sentimen.pkl')
    rf_aspek_model = joblib.load('rf_aspek_model.pkl')
    rf_sentimen_model = joblib.load('rf_sentimen_model.pkl')
    
    # Pastikan vektorizer sudah di-fit
    if not hasattr(tfidf_aspek, 'vocabulary_'):
        st.error("Vektorizer aspek belum di-fit dengan data.")
        st.stop()
    if not hasattr(tfidf_sentimen, 'vocabulary_'):
        st.error("Vektorizer sentimen belum di-fit dengan data.")
        st.stop()
except Exception as e:
    st.error(f"Gagal memuat model atau vektorizer: {e}")
    st.stop()

# Aplikasi Streamlit
def main():
    st.title("Sistem Prediksi Aspek dan Sentimen dengan Random Forest")
    st.markdown("### Sistem ini memprediksi:\n- **Aspek**: Fasilitas, Pelayanan, Masakan\n- **Sentimen**: Positif atau Negatif")
    
    # Pilihan input
    input_option = st.selectbox("Pilih input:", ["Teks", "File Excel"])
    
    if input_option == "Teks":
        user_input = st.text_area("Masukkan Teks", "")
        if st.button("Prediksi"):
            if not user_input:
                st.warning("Masukkan teks terlebih dahulu.")
            else:
                processed_text = preprocess_text(user_input, stopword_model, stemmer_model)
                aspect_vectorized = tfidf_aspek.transform([processed_text])
                predicted_aspect = rf_aspek_model.predict(aspect_vectorized)[0]
                
                if predicted_aspect == "tidak_dikenali":
                    st.write("**Aspek**: Tidak Dikenali")
                    st.write("**Sentimen**: -")
                else:
                    sentiment_vectorized = tfidf_sentimen.transform([processed_text])
                    predicted_sentiment = rf_sentimen_model.predict(sentiment_vectorized)[0]
                    st.write(f"**Aspek**: {predicted_aspect.capitalize()}")
                    st.write(f"**Sentimen**: {predicted_sentiment.capitalize()}")

    elif input_option == "File Excel":
        uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                if 'ulasan' not in df.columns:
                    st.error("File Excel harus memiliki kolom 'ulasan'.")
                    return
                
                results = {
                    "Aspek Tidak Dikenali": 0,
                    "Fasilitas": {"Positif": 0, "Negatif": 0},
                    "Pelayanan": {"Positif": 0, "Negatif": 0},
                    "Masakan": {"Positif": 0, "Negatif": 0}
                }
                
                for index, row in df.iterrows():
                    ulasan = row['ulasan']
                    processed_text = preprocess_text(ulasan, stopword_model, stemmer_model)
                    aspect_vectorized = tfidf_aspek.transform([processed_text])
                    predicted_aspect = rf_aspek_model.predict(aspect_vectorized)[0]
                    
                    if predicted_aspect == "tidak_dikenali":
                        results["Aspek Tidak Dikenali"] += 1
                    else:
                        sentiment_vectorized = tfidf_sentimen.transform([processed_text])
                        predicted_sentiment = rf_sentimen_model.predict(sentiment_vectorized)[0]
                        results[predicted_aspect.capitalize()][predicted_sentiment.capitalize()] += 1
                
                # Tampilkan hasil
                st.write("Hasil Prediksi:")
                st.write(results)
            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file Excel: {e}")

if __name__ == "__main__":
    main()
