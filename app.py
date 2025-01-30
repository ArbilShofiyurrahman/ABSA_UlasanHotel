import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
import io
import numpy as np
from io import BytesIO

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Menghapus angka dan karakter non-huruf
    return text

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
    
except Exception as e:
    st.error(f"Gagal memuat model atau vektorizer: {e}")
    st.stop()

def main():
    st.title("Sistem Prediksi Aspek dan Sentimen dengan Random Forest")
    st.markdown("### Sistem ini memprediksi:")
    st.markdown("- **Aspek**: Fasilitas, Pelayanan, Masakan")
    st.markdown("- **Sentimen**: Positif atau Negatif")
    
    # Input teks
    st.subheader("Input Teks Tunggal")
    user_input = st.text_area("Masukkan Teks", "")
    if st.button("Prediksi Teks"):
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

    # Input File Excel
    st.subheader("Input File Excel")
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            if 'ulasan' not in df.columns:
                st.error("File Excel harus memiliki kolom 'ulasan'.")
                return
            
            df["Aspek"] = ""
            df["Sentimen"] = ""
            total_rows = len(df)
            
            for index, row in df.iterrows():
                ulasan = str(row['ulasan'])
                processed_text = preprocess_text(ulasan, stopword_model, stemmer_model)
                aspect_vectorized = tfidf_aspek.transform([processed_text])
                predicted_aspect = rf_aspek_model.predict(aspect_vectorized)[0]

                if predicted_aspect == "tidak_dikenali":
                    df.at[index, "Aspek"] = "Tidak Dikenali"
                    df.at[index, "Sentimen"] = "-"
                else:
                    sentiment_vectorized = tfidf_sentimen.transform([processed_text])
                    predicted_sentiment = rf_sentimen_model.predict(sentiment_vectorized)[0]
                    df.at[index, "Aspek"] = predicted_aspect.capitalize()
                    df.at[index, "Sentimen"] = predicted_sentiment.capitalize()
            
            # Visualisasi Pie Chart
            st.subheader("Visualisasi Sentimen per Aspek")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            aspek_list = ["Fasilitas", "Pelayanan", "Masakan"]
            colors = ["#66b3ff", "#ff9999"]
            
            for i, aspek in enumerate(aspek_list):
                data = df[df['Aspek'] == aspek]['Sentimen'].value_counts()
                if not data.empty:
                    axes[i].pie(data, labels=data.index, autopct='%1.1f%%', colors=colors, startangle=140)
                    axes[i].set_title(f"Aspek {aspek}")
                else:
                    axes[i].pie([1], labels=["Tidak Ada Data"], colors=["#d3d3d3"])
                    axes[i].set_title(f"Aspek {aspek}")
            
            st.pyplot(fig)
            
            # Menampilkan DataFrame hasil prediksi
            st.subheader("Hasil Analisis")
            st.dataframe(df)
            
            # Download hasil
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
            output.seek(0)
            
            st.download_button(
                label="📥 Download Hasil Lengkap (Excel)",
                data=output,
                file_name="hasil_analisis_file.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file Excel: {e}")

if __name__ == "__main__":
    main()
