"""Preprocessing, feature extraction, dan model untuk aplikasi.
Kodenya disalin dari notebook (Preprocessing_Data dan Training_Model) supaya fitur dari aplikasi sama dengan saat training (KF-04).
"""
import io

import librosa
import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torch.nn as nn

SR, DUR = 16000, 6
N_SAMPLES = SR * DUR
N_FFT, HOP, N_MELS, N_MFCC, FMAX = 1024, 512, 64, 40, 3500
MIN_SEC, MAX_SEC = 1, 30  # batas durasi audio dari pengguna

CLASSES = ["burping", "needs", "physical_pain", "scared"]
LABELS = {"burping": "Perlu sendawa", "needs": "Kebutuhan umum", "physical_pain": "Sakit fisik", "scared": "Kaget/takut"}

HPF = scipy.signal.butter(4, 50, btype="highpass", fs=SR, output="sos")


# --- preprocessing dan fitur (sama dengan Preprocessing_Data) ---
def load_audio(data):
    y, _ = librosa.load(io.BytesIO(data), sr=SR, mono=True)  # dibaca dari memori, tidak ada file yang disimpan
    return y.astype(np.float32)


def preprocess(y):
    y = scipy.signal.sosfilt(HPF, y)  # buang noise frekuensi rendah (< 50 Hz)
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 0:
        y = y / peak
    if len(y) > N_SAMPLES:
        s = (len(y) - N_SAMPLES) // 2  # potong di tengah
        y = y[s:s + N_SAMPLES]
    else:
        y = np.pad(y, (0, N_SAMPLES - len(y)))  # zero padding di akhir
    return y.astype(np.float32)


def mel_db(y):
    return librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=FMAX, center=True), ref=1.0)


def features(y):
    """Sinyal 16 kHz mono -> matriks (104, 188) float16: 64 baris Mel + 40 baris MFCC."""
    M = mel_db(preprocess(y))
    return np.vstack([M, librosa.feature.mfcc(S=M, n_mfcc=N_MFCC)]).astype(np.float16)  # float16 seperti di X.npy


def sounding_steps(f):
    """Jumlah langkah waktu (dari 47) yang berbunyi. Frame zero padding punya nilai Mel minimum di semua baris."""
    mel = f[:N_MELS].astype(np.float32)
    ada = np.flatnonzero(mel.max(0) > mel.min() + 0.01)  # frame yang tidak kosong
    frame_akhir = ada[-1] + 1 if len(ada) else 0
    return max((frame_akhir + 3) // 4, 1)


# --- model (sama dengan Training_Model) ---
def conv_block(c_in, c_out, pool):
    return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(pool))


class CryNet(nn.Module):
    def __init__(self, bilstm=False, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(conv_block(1, 32, (2, 2)), conv_block(32, 64, (2, 2)), conv_block(64, 128, (2, 1)))
        self.bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True) if bilstm else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 4))

    def forward(self, x, n):               # x: (batch, 104, 188), n: jumlah langkah waktu yang berbunyi
        x = self.cnn(x.unsqueeze(1))       # (batch, 128, 13, 47)
        x = x.mean(2).transpose(1, 2)      # (batch, 47, 128)
        if self.bilstm is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, n.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = self.bilstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=47)
        ada = (torch.arange(x.shape[1], device=x.device) < n[:, None]).unsqueeze(2)
        return self.head((x * ada).sum(1) / n[:, None])


def load_model(path="outputs/model_final.pt"):
    ck = torch.load(path, map_location="cpu")  # selalu CPU, seperti di laptop pengguna
    model = CryNet(ck["bilstm"], ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["mean"], ck["std"]


# --- dipakai aplikasi ---
def check(name, data):
    """Pesan error kalau audio ditolak (KF-03), None kalau valid."""
    if not name.lower().endswith((".wav", ".mp3")):
        return "Format berkas tidak didukung. Gunakan WAV atau MP3."
    try:
        detik = sf.info(io.BytesIO(data)).duration
    except Exception:
        return "Berkas tidak dapat dibaca. Coba berkas lain."
    if not MIN_SEC <= detik <= MAX_SEC:
        return "Durasi audio harus antara 1 dan 30 detik."
    return None


@torch.no_grad()
def classify_features(model, mean, std, f):
    """Fitur (104, 188) -> 4 probabilitas yang jumlahnya 1."""
    x = (torch.from_numpy(f.astype(np.float32))[None] - mean) / std  # normalisasi dengan mean/std dari data train
    n = torch.tensor([sounding_steps(f)])
    return torch.softmax(model(x, n), 1)[0].numpy()


def classify(model, mean, std, data):
    return classify_features(model, mean, std, features(load_audio(data)))


def persen(p):
    """Persentase bulat yang jumlahnya tepat 100 (sisa pembulatan diberikan ke angka desimal terbesar)."""
    raw = np.asarray(p) * 100
    bulat = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - bulat))[:100 - bulat.sum()]:
        bulat[i] += 1
    return bulat
