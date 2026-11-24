"""
train.py — تدريب نموذج كشف الصوت المزيّف

التشغيل:
    python train.py              # تدريب + تقييم صادق (Cross-Validation)
    python train.py --no-cv      # تدريب سريع بدون تقييم

المدخلات:  dataset/REAL  و  dataset/FAKE   (wav / mp3 / flac / ogg)
المخرجات: model/voice_model.joblib  و  model/metrics.json
"""
import os
import re
import sys
import json
import time
import argparse
import warnings

import numpy as np
import joblib
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

import features as F

warnings.filterwarnings("ignore")

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}
AUG_SUFFIX = re.compile(r"_(fast|slow|noisy)$", re.IGNORECASE)
MODEL_PATH = os.path.join("model", "voice_model.joblib")
METRICS_PATH = os.path.join("model", "metrics.json")


# ----------------------------------------------------------
# 1) جمع الملفات — بدون تكرار
# ----------------------------------------------------------
def collect_files(folder):
    """
    يتجاهل:
      - النسخ المكبّرة القديمة (_fast / _slow / _noisy) لأن التكبير صار يصير داخل التدريب
      - نسخة mp3 إذا فيه wav بنفس الاسم (نفس الصوت مرتين = غش في التقييم)
    """
    by_stem = {}
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in AUDIO_EXT or AUG_SUFFIX.search(stem):
            continue
        if stem in by_stem and by_stem[stem].lower().endswith(".wav"):
            continue
        by_stem[stem] = name
    return [os.path.join(folder, n) for n in by_stem.values()]


def voice_group(path, label, idx):
    """
    نجمع ملفات نفس الصوت/المصدر في مجموعة وحدة، عشان التقييم يختبر النموذج
    على أصوات ما سمعها أبدًا (هذا الاختبار الحقيقي).
    """
    if label == 0:
        return f"real_{idx // 6}"
    b = os.path.basename(path)
    m = re.match(r"ElevenLabs_.*?T\d\d_\d\d_\d\d_(.+?)_(pvc|pre)_", b)
    if m:
        return "EL_" + m.group(1)
    if b.startswith("ElevenLabs"):
        return "EL_other"
    if b.lower().startswith("ttsmaker"):
        return "ttsmaker"
    if re.match(r"^\d+\.\w+$", b):
        return "numeric_ids"
    return "fake_" + b


# ----------------------------------------------------------
# 2) النموذج: تصويت بين 3 نماذج مختلفة
# ----------------------------------------------------------
def build_model():
    return VotingClassifier(
        estimators=[
            ("logreg", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))),
            ("svm", make_pipeline(StandardScaler(), SVC(C=3, gamma="scale", probability=True, random_state=0))),
            ("trees", ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=0)),
        ],
        voting="soft",
    )


def stack(feats, idx, keys):
    return np.concatenate([feats[k][idx] for k in keys])


def safe_extract(path, seed):
    try:
        return F.features_with_augmentations(path, seed)
    except Exception as e:
        print(f"  ⚠️ تم تجاهل {os.path.basename(path)}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--no-cv", action="store_true", help="تدريب بدون تقييم")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    real = collect_files(os.path.join(args.dataset, "REAL"))
    fake = collect_files(os.path.join(args.dataset, "FAKE"))
    files = real + fake
    labels = np.array([0] * len(real) + [1] * len(fake))
    print(f"📂 REAL: {len(real)} ملف   |   FAKE: {len(fake)} ملف  (بعد حذف التكرار والنسخ المكبّرة القديمة)")
    if len(real) < 5 or len(fake) < 5:
        sys.exit("❌ لازم يكون فيه 5 ملفات على الأقل في كل من REAL و FAKE")

    t = time.time()
    print("🔄 استخراج الخصائص (الأصلي + 5 نسخ معدّلة لكل ملف)...")
    res = Parallel(n_jobs=args.jobs)(delayed(safe_extract)(f, i) for i, f in enumerate(files))
    keep = [i for i, r in enumerate(res) if r is not None]
    files = [files[i] for i in keep]
    labels = labels[keep]
    res = [res[i] for i in keep]
    keys = list(res[0].keys())
    feats = {k: np.stack([r[k] for r in res]) for k in keys}
    print(f"✅ تم في {time.time() - t:.0f} ثانية — {len(files)} ملف × {len(keys)} نسخ × {F.N_FEATURES} خاصية")

    groups = np.array([voice_group(f, l, i) for i, (f, l) in enumerate(zip(files, labels))])
    metrics = {"n_real": int((labels == 0).sum()), "n_fake": int((labels == 1).sum())}

    # ------------------------------------------------------
    # 3) تقييم صادق: كل مرة نخبّي مجموعة أصوات كاملة عن النموذج
    # ------------------------------------------------------
    if not args.no_cv:
        n_splits = min(5, len(set(groups[labels == 1])), len(set(groups[labels == 0])))
        print(f"\n🧪 تقييم على أصوات ما شافها النموذج ({n_splits}-fold)...")
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        probs = {k: np.zeros(len(files)) for k in ("clean", "noise", "mp3", "phone")}
        for fold, (tr, te) in enumerate(cv.split(files, labels, groups), 1):
            m = build_model().fit(stack(feats, tr, keys), np.tile(labels[tr], len(keys)))
            for k in probs:
                probs[k][te] = m.predict_proba(feats[k][te])[:, 1]
            print(f"   fold {fold}: دقة = {accuracy_score(labels[te], probs['clean'][te] > 0.5) * 100:.1f}%")

        p = probs["clean"]
        tn, fp, fn, tp = confusion_matrix(labels, p > 0.5).ravel()
        metrics.update({
            "accuracy": float(accuracy_score(labels, p > 0.5)),
            "roc_auc": float(roc_auc_score(labels, p)),
            "accuracy_noisy": float(accuracy_score(labels, probs["noise"] > 0.5)),
            "accuracy_mp3": float(accuracy_score(labels, probs["mp3"] > 0.5)),
            "accuracy_phone": float(accuracy_score(labels, probs["phone"] > 0.5)),
            "confusion": {"real_ok": int(tn), "real_as_fake": int(fp), "fake_as_real": int(fn), "fake_ok": int(tp)},
        })
        print("\n📊 النتائج (على أصوات جديدة):")
        print(f"   الدقة:                 {metrics['accuracy'] * 100:.1f}%")
        print(f"   ROC-AUC:               {metrics['roc_auc']:.3f}")
        print(f"   الدقة مع ضوضاء:        {metrics['accuracy_noisy'] * 100:.1f}%")
        print(f"   الدقة بعد ضغط MP3:     {metrics['accuracy_mp3'] * 100:.1f}%")
        print(f"   الدقة بجودة مكالمة:    {metrics['accuracy_phone'] * 100:.1f}%")
        print(f"   حقيقي اعتُبر مزيّف: {fp}   |   مزيّف اعتُبر حقيقي: {fn}")
        wrong = [os.path.basename(files[i]) for i in np.where((p > 0.5) != labels)[0]]
        if wrong:
            print("   الملفات اللي غلط فيها:", ", ".join(wrong))

    # ------------------------------------------------------
    # 4) التدريب النهائي على كل البيانات
    # ------------------------------------------------------
    print("\n🚀 التدريب النهائي على كل البيانات...")
    model = build_model().fit(stack(feats, np.arange(len(files)), keys), np.tile(labels, len(keys)))
    os.makedirs("model", exist_ok=True)
    # إحصائيات "بصمة المتحدث" لاستخدامها في مقارنة صوتين
    from detector import _SPK_IDX
    spk = feats["clean"][:, _SPK_IDX]
    joblib.dump({
        "model": model, "n_features": F.N_FEATURES, "sample_rate": F.SR, "metrics": metrics,
        "spk_mean": spk.mean(axis=0), "spk_std": spk.std(axis=0),
    }, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"💾 تم حفظ النموذج في {MODEL_PATH}")


if __name__ == "__main__":
    main()
