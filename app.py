# Aplikasi web Pendeteksi Kebutuhan Bayi. Jalankan: streamlit run app.py
import hashlib
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st

import cry

st.set_page_config(page_title="Pendeteksi Kebutuhan Bayi", layout="wide")


# muat model sekali saat aplikasi dibuka (None kalau belum ada)
@st.cache_resource
def get_model():
    if not Path("outputs/model_final.pt").exists():
        return None
    m = cry.load_model()
    # warm-up dengan audio noise supaya analisis pertama tidak lambat
    buf = io.BytesIO()
    sf.write(buf, 0.1 * np.random.default_rng(0).standard_normal(cry.SR * 6), cry.SR, format="WAV")
    cry.classify(*m, buf.getvalue())
    return m


# judul dan dua kolom: input di kiri, hasil di kanan
model = get_model()
st.title("Pendeteksi Kebutuhan Bayi")
st.caption("Hasil hanya perkiraan dari pola suara tangisan, bukan diagnosis medis.")
kiri, kanan = st.columns(2)

# input audio: unggah file atau rekam
with kiri:
    sumber = st.radio("Sumber audio", ["Unggah File", "Rekam"], horizontal=True)
    if sumber == "Unggah File":
        audio = st.file_uploader("Berkas WAV atau MP3")  # format dicek di cry.check
        nama = audio.name if audio else None
    else:
        audio = st.audio_input("Rekam tangisan (3 sampai 10 detik)")
        nama = "rekaman.wav"
        st.caption("Jika mikrofon tidak bisa dipakai, pilih Unggah File.")
    data = audio.getvalue() if audio else None

    # validasi format dan durasi audio
    error = cry.check(nama, data) if data else None
    if error:
        st.error(error)
    elif data:
        st.audio(data, format="audio/mp3" if nama.lower().endswith(".mp3") else "audio/wav")
    if model is None:
        st.error("Model belum tersedia.")

    # tombol analisis, hasil disimpan bersama penanda audio-nya
    key = hashlib.md5(data).hexdigest() if data and not error else None  # penanda audio yang sedang dipilih
    if st.button("Analisis Tangisan", type="primary", disabled=key is None or model is None):
        try:
            with st.spinner("Menganalisis..."):
                st.session_state.hasil = (key, cry.classify(*model, data))
        except Exception:
            st.error("Audio gagal diproses. Coba lagi.")

# tampilan hasil: kelas teratas dan persentase tiap kelas
with kanan:
    st.subheader("Hasil")
    hasil = st.session_state.get("hasil")
    if hasil and hasil[0] == key:  # hasil lama hilang kalau audio diganti
        p = hasil[1]
        pct = cry.persen(p)
        atas = int(p.argmax())
        with st.container(border=True):
            st.markdown(f"## **{cry.LABELS[cry.CLASSES[atas]]} ({pct[atas]}%)**")
        for i in np.argsort(-p):
            teks = f"{cry.LABELS[cry.CLASSES[i]]}: {pct[i]}%"
            st.progress(int(pct[i]) / 100, text=f"**{teks}**" if i == atas else teks)
    else:
        st.info("Belum ada hasil.")
