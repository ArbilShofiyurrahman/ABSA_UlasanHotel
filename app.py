import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Fungsi Preprocessing
def preprocess_text(text):
    text = text.lower()  # Ubah ke huruf kecil
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hilangkan karakter non-alfabet
    text = re.sub(r'\s+', ' ', text)  # Hilangkan spasi berlebih
    return text

# Memuat Model
normalizer_model = joblib.load('normalizer_model.pkl')
stopword_model = joblib.load('stopword_model.pkl')
stemmer_model = joblib.load('stemmer_model.pkl')
vectorizers = {
    "aspek": joblib.load('tfidfaspek.pkl'),
    "fasilitas": joblib.load('tfidf_vectorizer_fasilitas.pkl'),
    "pelayanan": joblib.load('tfidf_vectorizer_pelayanan.pkl'),
    "masakan": joblib.load('tfidf_vectorizer_masakan.pkl')
}
aspect_model = joblib.load('aspek.pkl')
sentiment_models = {
    "fasilitas": joblib.load('model_random_forest_fasilitas.pkl'),
    "pelayanan": joblib.load('model_random_forest_pelayanan.pkl'),
    "masakan": joblib.load('model_random_forest_masakan.pkl')
}

# Aplikasi Streamlit
def main():
    st.title("Sistem Prediksi Aspek dan Sentimen dengan Random Forest")
    st.markdown("### Sistem ini memprediksi:\n- **Aspek**: Fasilitas, Pelayanan, Masakan\n- **Sentimen**: Positif atau Negatif")
    
    # Pilihan input
    input_option = st.selectbox("Pilih input:", ["Teks", "File Excel"])
    
    if input_option == "Teks":
        # Input teks dari pengguna
        user_input = st.text_area("Masukkan Teks", "")
        
        if st.button("Prediksi"):
            # Preprocessing
            processed_text = preprocess_text(user_input)
            
            # Prediksi Aspek
            aspect_vectorized = vectorizers["aspek"].transform([processed_text])
            predicted_aspect = aspect_model.predict(aspect_vectorized)[0]
            
            # Validasi aspek
            if predicted_aspect not in vectorizers:
                st.error("Aspek tidak dikenali.")
                return
            
            # Prediksi Sentimen
            sentiment_vectorizer = vectorizers[predicted_aspect]
            sentiment_model = sentiment_models[predicted_aspect]
            sentiment_vectorized = sentiment_vectorizer.transform([processed_text])
            predicted_sentiment = sentiment_model.predict(sentiment_vectorized)[0]
            
            # Menampilkan hasil prediksi
            st.write(f"**Aspek**: {predicted_aspect.capitalize()}")
            st.write(f"**Sentimen**: {predicted_sentiment.capitalize()}")

    elif input_option == "File Excel":
        # Upload file Excel
        uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
        
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            
            if 'ulasan' not in df.columns:
                st.error("File Excel harus memiliki kolom 'ulasan'.")
                return
            
            # Preprocessing dan prediksi untuk setiap baris dalam file
            results = {
                "Aspek Tidak Dikenali": 0,
                "Fasilitas": {"Positif": 0, "Negatif": 0},
                "Pelayanan": {"Positif": 0, "Negatif": 0},
                "Masakan": {"Positif": 0, "Negatif": 0}
            }
            
            for index, row in df.iterrows():
                ulasan = row['ulasan']
                processed_text = preprocess_text(ulasan)
                
                # Prediksi Aspek
                aspect_vectorized = vectorizers["aspek"].transform([processed_text])
                predicted_aspect = aspect_model.predict(aspect_vectorized)[0]
                
                if predicted_aspect not in vectorizers:
                    results["Aspek Tidak Dikenali"] += 1
                    continue  # Jika aspek tidak dikenali, lanjutkan ke ulasan berikutnya
                
                # Prediksi Sentimen
                sentiment_vectorizer = vectorizers[predicted_aspect]
                sentiment_model = sentiment_models[predicted_aspect]
                sentiment_vectorized = sentiment_vectorizer.transform([processed_text])
                predicted_sentiment = sentiment_model.predict(sentiment_vectorized)[0]
                
                if predicted_aspect == "fasilitas":
                    results["Fasilitas"][predicted_sentiment.capitalize()] += 1
                elif predicted_aspect == "pelayanan":
                    results["Pelayanan"][predicted_sentiment.capitalize()] += 1
                elif predicted_aspect == "masakan":
                    results["Masakan"][predicted_sentiment.capitalize()] += 1
            
            # Menampilkan hasil prediksi dalam bentuk diagram lingkaran
            aspect_names = ["Aspek Tidak Dikenali", "Fasilitas", "Pelayanan", "Masakan"]
            aspect_values = [
                results["Aspek Tidak Dikenali"],
                sum(results["Fasilitas"].values()),
                sum(results["Pelayanan"].values()),
                sum(results["Masakan"].values())
            ]
            
            # Plot lingkaran untuk distribusi aspek
            fig, ax = plt.subplots()
            ax.pie(aspect_values, labels=aspect_names, autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99'])
            ax.axis('equal')
            st.pyplot(fig)
            
            # Plot lingkaran untuk distribusi sentimen per aspek
            for aspect in ["Fasilitas", "Pelayanan", "Masakan"]:
                positive = results[aspect]["Positif"]
                negative = results[aspect]["Negatif"]
                fig, ax = plt.subplots()
                ax.pie([positive, negative], labels=["Positif", "Negatif"], autopct='%1.1f%%', startangle=90, colors=['#66B3FF', '#FF9999'])
                ax.axis('equal')
                st.write(f"Distribusi Sentimen untuk Aspek {aspect}:")
                st.pyplot(fig)

# Menjalankan aplikasi
if __name__ == "__main__":
    main()
