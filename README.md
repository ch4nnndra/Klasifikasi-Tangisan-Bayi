# Klasifikasi Tangisan Bayi

Aplikasi web yang memperkirakan kebutuhan bayi dari suara tangisan: Kebutuhan umum, Sakit fisik, Kaget/takut, atau Perlu sendawa. Model: CNN + BiLSTM (PyTorch), antarmuka: Streamlit.

## Menjalankan aplikasi
```bash
pip install -r requirements.txt
streamlit run app.py
```
`outputs/model_final.pt` sudah disertakan, jadi aplikasi bisa langsung dipakai.

## Mengulang dari awal
Dataset [Infant Cry Classification (4 Classes)](https://www.kaggle.com/datasets/karimeletriby/infant-cry-classification-4-classes) tidak ikut di repo. Taruh di `audio/<kelas>/*.wav`, lalu jalankan notebook berurutan:

1. `Audit_Data.ipynb`: cek kualitas data, buang duplikat dan klip yang labelnya bertentangan.
2. `Preprocessing_Data.ipynb`: split per grup rekaman, fitur Mel-Spectrogram + MFCC, augmentation.
3. `Training_Model.ipynb`: ablation Model A (CNN) vs Model B (CNN + BiLSTM), lalu simpan model final.

## Tes
```bash
python test_app.py
```
Mengecek validasi audio, kesamaan fitur aplikasi dengan fitur training, probabilitas, waktu respons, dan hasil test di notebook (butuh folder `audio/`).
