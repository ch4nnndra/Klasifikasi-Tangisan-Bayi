# Preprocessing, fitur, dan model untuk aplikasi (sama dengan notebook).
import io

import librosa
import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torch.nn as nn

# konstanta audio dan fitur (sama dengan notebook)
SR, DUR = 16000, 6
N_SAMPLES = SR * DUR
N_FFT, HOP, N_MELS, N_MFCC, FMAX = 1024, 512, 64, 40, 3500
MIN_SEC, MAX_SEC = 1, 30  # batas durasi audio dari pengguna

CLASSES = ["burping", "needs", "physical_pain", "scared"]
LABELS = {"burping": "Perlu sendawa", "needs": "Kebutuhan umum", "physical_pain": "Sakit fisik", "scared": "Kaget/takut"}

HPF = scipy.signal.butter(4, 50, btype="highpass", fs=SR, output="sos")  # filter high-pass 50 Hz


# baca audio dari bytes jadi mono 16 kHz
def load_audio(data):
    y, _ = librosa.load(io.BytesIO(data), sr=SR, mono=True)
    return y.astype(np.float32)


# high-pass, normalisasi, lalu potong atau padding ke 6 detik
def preprocess(y):
    y = scipy.signal.sosfilt(HPF, y)
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 0:
        y = y / peak
    if len(y) > N_SAMPLES:
        s = (len(y) - N_SAMPLES) // 2
        y = y[s:s + N_SAMPLES]
    else:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    return y.astype(np.float32)


# Mel-Spectrogram 64 band (0-3,5 kHz) dalam dB
def mel_db(y):
    return librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=FMAX, center=True), ref=1.0)


# fitur (104, 188): Mel + MFCC
def features(y):
    M = mel_db(preprocess(y))
    return np.vstack([M, librosa.feature.mfcc(S=M, n_mfcc=N_MFCC)]).astype(np.float16)


# jumlah langkah waktu yang berbunyi (n_all di notebook)
def sounding_steps(f):
    mel = f[:N_MELS].astype(np.float32)
    ada = np.flatnonzero(mel.max(0) > mel.min() + 0.01)
    frame_akhir = ada[-1] + 1 if len(ada) else 0
    return max((frame_akhir + 3) // 4, 1)


# satu blok CNN: Conv 3x3, BatchNorm, ReLU, MaxPool
def conv_block(c_in, c_out, pool):
    return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(pool))


# Model A (bilstm=False) dan Model B (bilstm=True), sama dengan notebook
class CryNet(nn.Module):
    def __init__(self, bilstm=False, dropout=0.3, units=64):
        super().__init__()
        self.cnn = nn.Sequential(conv_block(1, 32, (2, 2)), conv_block(32, 64, (2, 2)), conv_block(64, 128, (2, 1)))
        self.bilstm = nn.LSTM(128, units, batch_first=True, bidirectional=True) if bilstm else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * units if bilstm else 128, 4))

    # x: fitur (batch, baris, 188), n: panjang suara tiap klip
    def forward(self, x, n):
        x = self.cnn(x.unsqueeze(1))   # (batch, 128, baris / 8, 47)
        x = x.mean(2).transpose(1, 2)  # rata-rata frekuensi: (batch, 47, 128)
        if self.bilstm is not None:
            # BiLSTM hanya membaca langkah yang berbunyi
            x = nn.utils.rnn.pack_padded_sequence(x, n.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = self.bilstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=47)
        ada = (torch.arange(x.shape[1], device=x.device) < n[:, None]).unsqueeze(2)  # 1 untuk langkah yang berbunyi
        return self.head((x * ada).sum(1) / n[:, None])  # rata-rata waktu tanpa padding


# muat model, mean, std, dan T
def load_model(path="outputs/model_final.pt"):
    ck = torch.load(path, map_location="cpu")
    model = CryNet(ck["bilstm"], ck["dropout"], ck["units"])
    model.load_state_dict(ck["state_dict"])
    model.temperature = ck["temperature"]
    model.feat = slice(*ck["feat"])  # baris fitur yang dipakai model
    return model.eval(), ck["mean"], ck["std"]


# pesan error kalau audio ditolak, None kalau valid
def check(name, data):
    if not name.lower().endswith((".wav", ".mp3")):
        return "Format berkas tidak didukung. Gunakan WAV atau MP3."
    try:
        detik = sf.info(io.BytesIO(data)).duration
    except Exception:
        return "Berkas tidak dapat dibaca. Coba berkas lain."
    if not MIN_SEC <= detik <= MAX_SEC:
        return "Durasi audio harus antara 1 dan 30 detik."
    return None


# probabilitas 4 kelas (logit dibagi T)
@torch.no_grad()
def classify_features(model, mean, std, f):
    x = (torch.from_numpy(f[model.feat].astype(np.float32))[None] - mean) / std
    n = torch.tensor([sounding_steps(f)])
    return torch.softmax(model(x, n) / model.temperature, 1)[0].numpy()


# probabilitas dari bytes audio
def classify(model, mean, std, data):
    return classify_features(model, mean, std, features(load_audio(data)))


# persen bulat dengan jumlah 100
def persen(p):
    raw = np.asarray(p) * 100
    bulat = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - bulat))[:100 - bulat.sum()]:
        bulat[i] += 1
    return bulat
