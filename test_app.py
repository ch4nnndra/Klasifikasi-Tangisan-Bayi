# Tes untuk cry.py. Jalankan: python test_app.py (butuh folder audio/ dan hasil notebook di outputs/)
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
ck = torch.load("outputs/model_final.pt", map_location="cpu")
f1_notebook, T = ck["f1_test"], ck["temperature"]


# noise WAV sepanjang `detik`
def wav(detik, ext="WAV"):
    buf = io.BytesIO()
    sf.write(buf, 0.1 * np.random.default_rng(0).standard_normal(int(cry.SR * detik)), cry.SR, format=ext)
    return buf.getvalue()


# validasi audio
assert cry.check("a.txt", b"halo") == "Format berkas tidak didukung. Gunakan WAV atau MP3."
assert cry.check("a.wav", b"bukan audio") == "Berkas tidak dapat dibaca. Coba berkas lain."
assert cry.check("a.wav", wav(0.5)) == "Durasi audio harus antara 1 dan 30 detik."
assert cry.check("a.wav", wav(40)) == "Durasi audio harus antara 1 dan 30 detik."
assert cry.check("a.wav", wav(3)) is None
try:
    assert cry.check("a.mp3", wav(3, "MP3")) is None and len(cry.classify(model, mean, std, wav(3, "MP3"))) == 4
    print("MP3: bisa diproses")
except Exception as e:  # libsndfile tanpa MP3
    print("MP3 dilewati:", e)
print("Validasi audio: lulus")

# persen selalu berjumlah 100
for p in np.random.default_rng(0).dirichlet(np.ones(4), 200):
    assert cry.persen(p).sum() == 100
print("Persentase: jumlah selalu 100")

# sounding_steps sama dengan n_all di notebook
mel = torch.from_numpy(X[:, :64]).float()
kosong = mel.amax(1) <= mel.amin((1, 2))[:, None] + 0.01
akhir = torch.where(~kosong, torch.arange(188), -1).amax(1) + 1
n_nb = ((akhir + 3) // 4).clamp(min=1).numpy()
assert (n_nb == np.array([cry.sounding_steps(f) for f in X])).all()
print(f"Langkah berbunyi: sama ({len(X)} klip)")

# fitur aplikasi sama dengan X.npy
contoh = d.sample(10, random_state=0)
for i, p in zip(contoh.index, contoh.path):
    f = cry.features(cry.load_audio(Path(p).read_bytes()))
    assert np.array_equal(f, X[i]), f"fitur beda di {p}, selisih maks {np.abs(f.astype(float) - X[i]).max()}"
print("Fitur: sama (10 klip)")

# 4 probabilitas dengan jumlah 1
for p in contoh.path:
    pr = cry.classify(model, mean, std, Path(p).read_bytes())
    assert pr.shape == (4,) and abs(pr.sum() - 1) < 1e-5 and (pr >= 0).all()
print("Probabilitas: 4 kelas, jumlah 1")

# F1 test aplikasi sama dengan notebook
te = d.index[d.split == "test"]
pred = [cry.classify_features(model, mean, std, X[i]).argmax() for i in te]
f1_app = f1_score(d.y[te], pred, average="macro")
print(f"F1-Macro test: aplikasi {f1_app:.4f}, notebook {f1_notebook:.4f}")
assert abs(f1_app - f1_notebook) < 0.02


# T tidak mengubah tebakan dan menurunkan NLL
def probabilitas(T):
    model.temperature = T
    return np.array([cry.classify_features(model, mean, std, X[i]) for i in te])


p_T, p_1 = probabilitas(T), probabilitas(1.0)
model.temperature = T
assert (p_T.argmax(1) == p_1.argmax(1)).all()
nll = lambda p: -np.log(np.clip(p[np.arange(len(te)), d.y[te].values], 1e-12, 1)).mean()
print(f"T = {T:.2f}: NLL {nll(p_1):.3f} ke {nll(p_T):.3f}, confidence {p_1.max(1).mean():.3f} ke {p_T.max(1).mean():.3f}")
assert nll(p_T) <= nll(p_1)

# waktu respons (median 20 kali)
data = Path(contoh.path.iloc[0]).read_bytes()
cry.classify(model, mean, std, data)  # warm-up
waktu = []
for _ in range(20):
    t = time.perf_counter()
    cry.classify(model, mean, std, data)
    waktu.append(time.perf_counter() - t)
print(f"Waktu respons: {statistics.median(waktu):.2f} detik")
assert statistics.median(waktu) <= 5


# analisis tidak menyimpan file
def snapshot():
    return {str(p): p.stat().st_size for p in Path(".").rglob("*")
            if p.is_file() and not {".venv", ".git", "__pycache__", "audio"} & set(p.parts)}


sebelum = snapshot()
cry.classify(model, mean, std, data)
assert snapshot() == sebelum
print("Penyimpanan: tidak ada file baru")
print("Semua tes lulus")
