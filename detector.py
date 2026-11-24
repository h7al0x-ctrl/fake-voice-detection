"""
detector.py - Model loading, prediction and voice comparison (used by app.py and predict.py).
"""
import os
import numpy as np
import joblib

import features as F

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "voice_model.joblib")

# Probabilities inside this range are reported as uncertain
UNSURE_LOW, UNSURE_HIGH = 0.35, 0.65

# Indices of the speaker-embedding features: MFCC 1-19 mean and std, plus pitch mean and std
_SPK_IDX = np.r_[1:20, 21:40, 224, 225]


class Detector:
    def __init__(self, path=MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}. Run `python train.py` first.")
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.metrics = bundle.get("metrics", {})
        self.spk_mean = bundle.get("spk_mean")
        self.spk_std = bundle.get("spk_std")
        if bundle.get("n_features") != F.N_FEATURES:
            raise RuntimeError("Model was trained with a different feature set. Run `python train.py` again.")

    # ---------------- Fake voice detection ----------------
    def analyze_features(self, feats):
        p_fake = float(self.model.predict_proba(feats.reshape(1, -1))[0, 1])
        label = "FAKE" if p_fake >= 0.5 else "REAL"
        confidence = p_fake if label == "FAKE" else 1 - p_fake
        return {
            "result": label,
            "fake_probability": round(p_fake * 100, 1),
            "confidence": round(confidence * 100, 1),
            "uncertain": UNSURE_LOW < p_fake < UNSURE_HIGH,
        }

    def analyze(self, path):
        y = F.load_audio(path)
        out = self.analyze_features(F.features_from_signal(y))
        out["duration"] = round(len(y) / F.SR, 2)
        return out

    # ---------------- Voice comparison ----------------
    def speaker_embedding(self, feats):
        e = feats[_SPK_IDX]
        if self.spk_mean is not None:
            e = (e - self.spk_mean) / (self.spk_std + 1e-8)
        return e

    def compare(self, path1, path2):
        f1 = F.extract_features(path1)
        f2 = F.extract_features(path2)
        e1, e2 = self.speaker_embedding(f1), self.speaker_embedding(f2)
        cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        similarity = (cos + 1) / 2 * 100  # map [-1, 1] to [0, 100]
        return {
            "similarity": round(similarity, 1),
            "same_speaker_likely": cos > 0.5,
            "voice1": self.analyze_features(f1),
            "voice2": self.analyze_features(f2),
        }
