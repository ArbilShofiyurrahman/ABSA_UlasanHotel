import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt

# Fungsi Preprocessing
def preprocess_text(text, normalizer_model, stopword_model, stemmer_model):
    text = text.lower()  # Casefolding
    text = normalizer_model(text)  # Normalisasi
    text = stopword_model.remove(text)  # Stopword Removal
    text = stemmer_model.stem(text)  # Stemming
    return text

# Memuat Model
normalizer_model = joblib.load('normalisasi_pld.pkl')
stopword_model = joblib.load('stopword_remover_model_pld.pkl')
stemmer_model = joblib.load('stemmer_model_pld.pkl')
tfidf_vectorizer_aspek = joblib.load('tfidf_vectorizer_aspek_pld.pkl')
rf_aspek_model = joblib.load('rf_aspek_model_pld.pkl')
rf_sentimen_model = joblib.load('rf_sentimen_model_pld.pkl')

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
            processed_text = preprocess_text(user_input, normalizer_model, stopword_model, stemmer_model)
            
            # TF-IDF untuk aspek
            aspect_vectorized = tfidf_vectorizer_aspek.transform([processed_text])
            
            # Prediksi Aspek
            predicted_aspect = rf_aspek_model.predict(aspect_vectorized)[0]
            
            if predicted_aspect == "tidak_dikenali":
                st.write("**Aspek**: Tidak Dikenali")
                st.write("**Sentimen**: -")
            else:
                # Prediksi Sentimen
                sentiment_vectorized = tfidf_vectorizer_aspek.transform([processed_text])
                predicted_sentiment = rf_sentimen_model.predict(sentiment_vectorized)[0]
                
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
                processed_text = preprocess_text(ulasan, normalizer_model, stopword_model, stemmer_model)
                
                # TF-IDF untuk aspek
                aspect_vectorized = tfidf_vectorizer_aspek.transform([processed_text])
                
                # Prediksi Aspek
                predicted_aspect = rf_aspek_model.predict(aspect_vectorized)[0]
                
                if predicted_aspect == "tidak_dikenali":
                    results["Aspek Tidak Dikenali"] += 1
                else:
                    # Prediksi Sentimen
                    sentiment_vectorized = tfidf_vectorizer_aspek.transform([processed_text])
                    predicted_sentiment = rf_sentimen_model.predict(sentiment_vectorized)[0]
                    
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
