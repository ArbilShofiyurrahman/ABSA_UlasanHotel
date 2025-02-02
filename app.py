import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
# Load icons
icons = {
    "Dashboard": "📊",
    "Analisis Data": "📑",
    "Report": "📋",
    "Testing": "🧪"
}

# Sidebar Navigation
st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih Menu", list(icons.keys()), format_func=lambda x: f"{icons[x]} {x}")


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
        r'\btidak memenuhi harapan\b': 'kecewa',
        r'\btidak jelek\b': 'bagus'
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

# Dashboard
if menu == "Dashboard":
    st.title("📊 Analisis Sentimen")
    st.write("Selamat datang di sistem analisis sentimen berbasis aspek untuk ulasan hotel.")
    st.markdown("""
    1. Positif : Ulasan yang mengandung kata-kata atau frasa yang menunjukkan kepuasan, pujian, atau pengalaman baik terhadap aspek tertentu (fasilitas, pelayanan, atau masakan).
    2. Negatif : Ulasan yang mengandung kata-kata atau frasa yang menunjukkan ketidakpuasan, kritik, atau pengalaman buruk terhadap aspek tertentu (fasilitas, pelayanan, atau masakan).
    """)
   

# Analisis Data
elif menu == "Analisis Data":
    st.title("📑 Analisis Data")
    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.write("### Data Awal")
            st.dataframe(df.head())

            if 'ulasan' in df.columns:
                df['Cleaned_Ulasan'] = df['Ulasan'].apply(lambda x: clean_text(str(x)))
                st.write("### Data Setelah Cleaning")
                st.dataframe(df[['ulasan', 'Cleaned_Ulasan']].head())

                # Simpan ke Session State untuk digunakan di Report
                st.session_state['processed_data'] = df
            else:
                st.error("Kolom 'ulasan' tidak ditemukan dalam dataset.")
        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}")


## Report
elif menu == "Report":
    st.title("📋 Report Hasil Analisis")
    if 'processed_data' in st.session_state:
        df = st.session_state['processed_data']
        st.dataframe(df[['Ulasan', 'Cleaned_Ulasan']].head())

        # Unduh hasil analisis
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Hasil Analisis')
            writer.close()
        output.seek(0)

        st.download_button("⬇️ Download Hasil Analisis", data=output, file_name="Hasil_Analisis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("⚠️ Tidak ada data untuk ditampilkan. Silakan lakukan analisis terlebih dahulu.")

# Testing
elif menu == "Testing":
    st.title("🧪 Pengujian Model")
    user_input = st.text_area("Masukkan teks ulasan hotel")

    if st.button("Prediksi"):
        if user_input:
            cleaned_text = clean_text(user_input)
            normalized_text = normalize_negation(cleaned_text)
            aspect_vectorized = tfidf_aspek.transform([normalized_text])
            sentiment_vectorized = tfidf_sentimen.transform([normalized_text])

            try:
                predicted_aspect = rf_aspek_model.predict(aspect_vectorized)[0]
                predicted_sentiment = rf_sentimen_model.predict(sentiment_vectorized)[0]

                st.success(f"**Aspek**: {predicted_aspect.capitalize()}")
                st.success(f"**Sentimen**: {predicted_sentiment.capitalize()}")
            except Exception as e:
                st.error(f"❌ Error dalam prediksi: {e}")
        else:
            st.warning("⚠️ Masukkan teks terlebih dahulu.")
