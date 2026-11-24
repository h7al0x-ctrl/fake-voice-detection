"""
predict.py — فحص ملف صوتي أو مجلد كامل من سطر الأوامر

أمثلة:
    python predict.py voice.wav
    python predict.py my_folder/ --csv results.csv
"""
import os
import csv
import sys
import argparse
from datetime import datetime

from detector import Detector

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}


def main():
    ap = argparse.ArgumentParser(description="كشف الصوت المزيّف")
    ap.add_argument("path", help="ملف صوتي أو مجلد")
    ap.add_argument("--csv", default="results.csv", help="ملف حفظ النتائج")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(os.path.join(args.path, f) for f in os.listdir(args.path)
                       if os.path.splitext(f)[1].lower() in AUDIO_EXT)
    elif os.path.isfile(args.path):
        files = [args.path]
    else:
        sys.exit(f"❌ ما لقيت {args.path}")

    det = Detector()
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "file", "result", "fake_probability_%", "confidence_%", "uncertain"])
        for path in files:
            try:
                r = det.analyze(path)
            except Exception as e:
                print(f"⚠️  {os.path.basename(path)}: {e}")
                continue
            note = "  (غير متأكد)" if r["uncertain"] else ""
            icon = "🔴" if r["result"] == "FAKE" else "🟢"
            print(f"{icon} {os.path.basename(path)}: {r['result']}  ثقة {r['confidence']}%{note}")
            w.writerow([datetime.now().isoformat(timespec="seconds"), path, r["result"],
                        r["fake_probability"], r["confidence"], r["uncertain"]])
    print(f"💾 النتائج محفوظة في {args.csv}")


if __name__ == "__main__":
    main()
