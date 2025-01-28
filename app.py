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

    # Lakukan prediksi aspek dan sentimen
                for ulasan in df['ulasan']:
                    aspek, sentimen = predict_aspect_and_sentiment(ulasan)
                    if aspek in results:
                        results[aspek][sentimen] += 1
                    else:
                        results["Aspek Tidak Dikenali"] += 1

                # Tampilkan hasil dalam tabel
                st.write("### Hasil Analisis Sentimen")
                hasil_df = pd.DataFrame(results).T
                st.dataframe(hasil_df)

                # Tampilkan pie chart dalam satu baris
                st.write("### Distribusi Sentimen per Aspek")
                columns = st.columns(len(results) - 1)  # Hapus "Aspek Tidak Dikenali"

                for i, (aspek, nilai) in enumerate(results.items()):
                    if aspek == "Aspek Tidak Dikenali":
                        continue  # Skip aspek tidak dikenali

                    labels = ["Positif", "Negatif"]
                    sizes = [nilai["Positif"], nilai["Negatif"]]
                    colors = ["#4CAF50", "#FF5252"]  # Hijau untuk positif, merah untuk negatif

                    fig, ax = plt.subplots(figsize=(3, 3))  # Pie chart lebih kecil
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.set_title(aspek, fontsize=12)

                    # Tampilkan di dalam kolom yang sesuai
                    with columns[i]:
                        st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file Excel: {e}")

if __name__ == "__main__":
    main()
