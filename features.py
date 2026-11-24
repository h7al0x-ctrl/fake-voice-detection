"""
features.py - Audio loading, feature extraction and training-time augmentation.

Training, the CLI and the web app all use this module, so features are always
computed the same way at training and inference time.
"""
import warnings

import numpy as np
import librosa
import scipy.fftpack

SR = 16000          # all audio is resampled to 16 kHz so the model cannot learn from recording bandwidth
N_FFT = 512
HOP = 160           # 10ms
N_MFCC = 20
N_LFCC = 20
TOP_DB = 30         # silence threshold for trimming leading/trailing silence
MIN_SECONDS = 0.5


def load_audio(path_or_file, sr=SR):
    """Load audio as 16 kHz mono, trim silence and peak-normalize."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(path_or_file, sr=sr, mono=True)
    except Exception:
        raise ValueError("Could not read this audio file. It may be damaged or in an unsupported format.")
    return prepare_signal(y, sr)


def prepare_signal(y, sr=SR):
    if y.size == 0:
        raise ValueError("The audio file is empty")
    if np.max(np.abs(y)) < 1e-4:  # digital silence (very quiet recordings are still accepted)
        raise ValueError("No sound detected — the recording is silent or almost silent")
    y_trim, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if len(y_trim) >= int(MIN_SECONDS * sr):
        y = y_trim
    if len(y) < int(MIN_SECONDS * sr):
        raise ValueError("The audio is too short (less than half a second of sound)")
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.95
    return y.astype(np.float32)


def _linear_filterbank(sr, n_fft, n_filters):
    """Linearly spaced triangular filterbank used for LFCC features."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = np.linspace(0, sr / 2, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb


_LFB = _linear_filterbank(SR, N_FFT, 40)


def _stats(m):
    """Mean and standard deviation of each row over time."""
    m = np.atleast_2d(m)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def features_from_signal(y, sr=SR):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2

    # MFCC with first and second order deltas
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=40)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
    d1 = librosa.feature.delta(mfcc, width=5)
    d2 = librosa.feature.delta(mfcc, order=2, width=5)

    # LFCC with first order deltas
    lfb = np.log(_LFB @ S + 1e-10)
    lfcc = scipy.fftpack.dct(lfb, axis=0, norm="ortho")[:N_LFCC]
    ld1 = librosa.feature.delta(lfcc, width=5)

    # spectral descriptors
    centroid = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=np.sqrt(S), sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=sr)
    flatness = librosa.feature.spectral_flatness(S=np.sqrt(S))
    contrast = librosa.feature.spectral_contrast(S=np.sqrt(S), sr=sr, n_bands=5, fmin=100)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(S=np.sqrt(S), frame_length=N_FFT)
    rms_db = librosa.amplitude_to_db(rms + 1e-6)

    # pitch statistics: synthetic speech often has smoother, more stable pitch
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=HOP)
    lf0 = np.log(f0)
    df0 = np.abs(np.diff(lf0))
    pitch = np.array([lf0.mean(), lf0.std(), np.median(df0), df0.mean()])

    return np.concatenate([
        _stats(mfcc), _stats(d1), _stats(d2),
        _stats(lfcc), _stats(ld1),
        _stats(centroid / 1000), _stats(bandwidth / 1000), _stats(rolloff / 1000),
        _stats(np.log(flatness + 1e-10)), _stats(contrast), _stats(zcr), _stats(rms_db),
        pitch,
    ]).astype(np.float32)


def extract_features(path_or_file):
    return features_from_signal(load_audio(path_or_file))


N_FEATURES = 228  # features per audio clip


# ======================================================
# Data augmentation (training only)
# ======================================================
def _noise(y, seed):
    r = np.random.RandomState(seed)
    snr = 20 + (seed % 3) * 5
    return y + r.randn(len(y)).astype(np.float32) * np.sqrt(np.mean(y ** 2) / 10 ** (snr / 10))


def _speed(y, s):
    return librosa.resample(y, orig_sr=SR, target_sr=int(SR / s))


def _mp3(y):
    """MP3 round-trip, simulating compressed audio from messaging apps or the web."""
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="MP3", compression_level=0.9)
    buf.seek(0)
    z, _ = sf.read(buf, dtype="float32")
    return z


def _phone(y):
    """Telephone-band low-pass filter (3.4 kHz)."""
    import scipy.signal as ss
    b, a = ss.butter(6, 3400 / (SR / 2))
    return ss.filtfilt(b, a, y)


AUGMENTATIONS = {
    "noise": lambda y, i: _noise(y, i),
    "fast": lambda y, i: _speed(y, 1.1),
    "slow": lambda y, i: _speed(y, 0.9),
    "mp3": lambda y, i: _mp3(y),
    "phone": lambda y, i: _phone(y),
}


def features_with_augmentations(path, seed=0):
    """Return features for the clean clip and every augmented version."""
    raw, _ = librosa.load(path, sr=SR, mono=True)
    out = {"clean": features_from_signal(prepare_signal(raw))}
    for name, fn in AUGMENTATIONS.items():
        out[name] = features_from_signal(prepare_signal(np.asarray(fn(raw, seed), dtype=np.float32)))
    return out
