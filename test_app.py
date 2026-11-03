"""Tes mandiri untuk cry.py. Jalankan: python test_app.py (butuh folder audio/ dan hasil notebook di outputs/)."""
import io
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import f1_score

import cry

d = pd.read_csv("outputs/data_siap.csv")
X = np.load("outputs/X.npy")
model, mean, std = cry.load_model()
f1_notebook = torch.load("outputs/model_final.pt", map_location="cpu")["f1_test"]


def wav(detik, ext="WAV"):
    buf = io.BytesIO()
    sf.write(buf, 0.1 * np.random.default_rng(0).standard_normal(int(cry.SR * detik)), cry.SR, format=ext)
    return buf.getvalue()


# KF-03: audio yang tidak valid ditolak dengan pesan yang sesuai
assert cry.check("a.txt", b"halo") == "Format berkas tidak didukung. Gunakan WAV atau MP3."
assert cry.check("a.wav", b"bukan audio") == "Berkas tidak dapat dibaca. Coba berkas lain."
assert cry.check("a.wav", wav(0.5)) == "Durasi audio harus antara 1 dan 30 detik."
assert cry.check("a.wav", wav(40)) == "Durasi audio harus antara 1 dan 30 detik."
assert cry.check("a.wav", wav(3)) is None
try:
    assert cry.check("a.mp3", wav(3, "MP3")) is None and len(cry.classify(model, mean, std, wav(3, "MP3"))) == 4
    print("MP3 diterima dan bisa diproses")
except Exception as e:  # libsndfile tanpa MP3
    print("MP3 dilewati:", e)
print("KF-03 lulus")

# persen: selalu 4 angka bulat dengan jumlah 100
for p in np.random.default_rng(0).dirichlet(np.ones(4), 200):
    assert cry.persen(p).sum() == 100
print("persen lulus (jumlah selalu 100)")

# sounding_steps (numpy) harus sama dengan perhitungan n_all di notebook (torch)
mel = torch.from_numpy(X[:, :64]).float()
kosong = mel.amax(1) <= mel.amin((1, 2))[:, None] + 0.01
akhir = torch.where(~kosong, torch.arange(188), -1).amax(1) + 1
n_nb = ((akhir + 3) // 4).clamp(min=1).numpy()
assert (n_nb == np.array([cry.sounding_steps(f) for f in X])).all()
print(f"sounding_steps sama dengan notebook untuk semua {len(X)} klip")

# KF-04: fitur dari aplikasi sama dengan fitur dari proses training (10 klip)
contoh = d.sample(10, random_state=0)
for i, p in zip(contoh.index, contoh.path):
    f = cry.features(cry.load_audio(Path(p).read_bytes()))
    assert np.array_equal(f, X[i]), f"fitur beda di {p}, selisih maks {np.abs(f.astype(float) - X[i]).max()}"
print("KF-04 lulus: fitur aplikasi sama persis dengan X.npy pada 10 klip")

# KF-05: 4 probabilitas, jumlahnya 1
for p in contoh.path:
    pr = cry.classify(model, mean, std, Path(p).read_bytes())
    assert pr.shape == (4,) and abs(pr.sum() - 1) < 1e-5 and (pr >= 0).all()
print("KF-05 lulus")

# jalur aplikasi (CPU, fp32) harus mengulang hasil test di notebook (GPU, bf16)
te = d.index[d.split == "test"]
pred = [cry.classify_features(model, mean, std, X[i]).argmax() for i in te]
f1_app = f1_score(d.y[te], pred, average="macro")
print(f"F1-Macro test: aplikasi {f1_app:.4f} | notebook {f1_notebook:.4f}")
assert abs(f1_app - f1_notebook) < 0.02

# KNF-01: waktu respons (nilai tengah 20 kali) di CPU untuk klip 6 detik
data = Path(contoh.path.iloc[0]).read_bytes()
cry.classify(model, mean, std, data)  # warm-up
waktu = []
for _ in range(20):
    t = time.perf_counter()
    cry.classify(model, mean, std, data)
    waktu.append(time.perf_counter() - t)
print(f"KNF-01: nilai tengah {statistics.median(waktu):.2f} detik (batas 5 detik)")
assert statistics.median(waktu) <= 5


# KNF-04: tidak ada file baru di folder aplikasi setelah analisis
def snapshot():
    return {str(p): p.stat().st_size for p in Path(".").rglob("*")
            if p.is_file() and not {".venv", ".git", "__pycache__", "audio"} & set(p.parts)}


sebelum = snapshot()
cry.classify(model, mean, std, data)
assert snapshot() == sebelum
print("KNF-04 lulus: tidak ada file yang tersimpan")
print("SEMUA TES LULUS")
