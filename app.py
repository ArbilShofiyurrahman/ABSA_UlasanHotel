import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt

def clean_text(text):
# Menghapus angka, tanda baca, dan karakter spesial
    text = re.sub(r'[^a-zA-Z\s]', '', text)
return text

# Fungsi Tokenisasi
def tokenize_text(text):
    # Menggunakan nltk untuk tokenisasi
    tokens = word_tokenize(text)
    return tokens
# Fungsi Normalisasi (Contoh)
def normalize_negation(text):
    negation_patterns = {
        r'\btidak bersih\b': 'kotor',
        r'\btidak teratur\b': 'berantakan',
        r'\btidak lengkap\b': 'tidaklengkap',
        r'\btidak memadai\b': 'kurangmemadai',
        r'\btidak nyaman\b': 'tidaknyaman',
        r'\btidak ramah\b': 'tidakramah',
        r'\btidak segar\b': 'tidaksegar',
        r'\btidak enak\b': 'tidakenak',
        r'\btidak sopan\b': 'tidaksopan',
        r'\btidak profesional\b': 'tidakprofesional',
        r'\btidak responsif\b': 'cuek',
        r'\btidak efisien\b': 'tidakefisien',
        r'\btidak konsisten\b': 'tidakkonsisten',
        r'\btidak stabil\b': 'tidakstabil',
        r'\btidak matang\b': 'tidakmatang',
        r'\btidak membantu\b': 'tidakmembantu',
        r'\btidak cepat\b': 'lambat',
        r'\btidak wajar\b': 'aneh',
        r'\btidak sesuai\b': 'tidaksesuai',
        r'\btidak aman\b': 'tidakaman',
        r'\btidak jujur\b': 'tidakjujur',
        r'\btidak peduli\b': 'cuek',
        r'\btidak terawat\b': 'tidakterawat',
        r'\btidak tepat waktu\b': 'tidaktepatwaktu',
        r'\btidak tanggap\b': 'tidaksigap',
        r'\btidak bertanggung jawab\b': 'tidakbertanggungjawab',
        r'\btidak wangi\b': 'bau',
        r'\btidak layak\b': 'tidaklayak',
       # Kata negasi diawali dengan "kurang"
        r'\bkurang bersih\b': 'kotor',
        r'\bkurang memuaskan\b': 'tidakmemuaskan',
        r'\bkurang sopan\b': 'tidaksopan',
        r'\bkurang cepat\b': 'lambat',
        r'\bkurang nyaman\b': 'tidaknyaman',
        r'\bkurang ramah\b': 'tidakramah',
        r'\bkurang segar\b': 'tidaksegar',
        r'\bkurang profesional\b': 'tidakprofesional',
        r'\bkurang terawat\b': 'tidakterawat',
        r'\bkurang efisien\b': 'tidakefisien',
        r'\bkurang matang\b': 'tidakmatang',
        r'\bkurang sigap\b': 'tidaksigap',
        r'\bkurang informatif\b': 'tidakinformatif',
        r'\bkurang sesuai ekspektasi\b': 'kecewa',
        # Slang dan typo
        r'\btdk bersih\b': 'kotor',
        r'\btdk cepat\b': 'lambat',
        r'\btdk nyaman\b': 'tidaknyaman',
        r'\btdk ramah\b': 'tidakramah',
        r'\bg enak\b': 'tidakenak',
        r'\bg aman\b': 'tidakaman',
        r'\bg sopan\b': 'tidaksopan',
        r'\bg stabil\b': 'tidakstabil',
        r'\btdk rapi\b': 'berantakan',
        r'\bg rapi\b': 'berantakan',
    
        # Frase tambahan sesuai konteks ulasan
        r'\btidak dilayani\b': 'cuek',
        r'\btdk dilayani\b': 'cuek',
        r'\btdk sesuai\b': 'kecewa',
        r'\btidak sesuai\b': 'kecewa',
        r'\btidak diprioritaskan\b': 'diabaikan',
        r'\btidak sesuai ekspektasi\b': 'kecewa',
        r'\btidak jujur\b': 'tidakjujur',
        r'\btdk jujur\b': 'tidakjujur',
        r'\btidak menepati janji\b': 'tidakjujur',
        r'\bkurang tanggung jawab\b': 'tidakbertanggungjawab',
        r'\bkurang perhatian\b': 'cuek',
        r'\bkurang detail\b': 'tidakdetail',
        r'\bkurang terorganisir\b': 'asal-asalan',
        r'\btidak terlaksana dengan baik\b': 'berantakan',
        r'\btidak memenuhi harapan\b': 'kecewa'
        }

    # Lakukan penggantian kata sesuai pola
    for pattern, replacement in negation_patterns.items():
        text = re.sub(pattern, replacement, text)
    return text


# Fungsi Preprocessing
def preprocess_text(text, normalizer_model, stopword_model, stemmer_model):
    text = text.lower()  # Casefolding
    text = clean_text(text) 
    text = normalize_negation(text)   # Normalisasi
    text = stopword_model.remove(text)  # Stopword Removal
    tokens = tokenize_text(text) 
    text = stemmer_model.stem(text)  # Stemming
    return text

# Memuat Model
stopword_model = joblib.load('stopword_remover_model.pkl')
stemmer_model = joblib.load('stemmer_model.pkl')
tfidf_vectorizer_aspek = joblib.load('tfidf_vectorizer_aspek.pkl')
tfidf_vectorizer_sentimen = joblib.load('tfidf_vectorizer_sentimen.pkl')
rf_aspek_model = joblib.load('rf_aspek_model.pkl')
rf_sentimen_model = joblib.load('rf_sentimen_model.pkl')

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
