"""
predict.py - Classify an audio file or a folder of audio files from the command line.

Examples:
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
    ap = argparse.ArgumentParser(description="Fake voice detection")
    ap.add_argument("path", help="Audio file or folder")
    ap.add_argument("--csv", default="results.csv", help="CSV file to append results to")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(os.path.join(args.path, f) for f in os.listdir(args.path)
                       if os.path.splitext(f)[1].lower() in AUDIO_EXT)
    elif os.path.isfile(args.path):
        files = [args.path]
    else:
        sys.exit(f"Not found: {args.path}")

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
                print(f"SKIPPED {os.path.basename(path)}: {e}")
                continue
            note = "  (uncertain)" if r["uncertain"] else ""
            print(f"{r['result']:4s}  {r['confidence']:5.1f}%  {os.path.basename(path)}{note}")
            w.writerow([datetime.now().isoformat(timespec="seconds"), path, r["result"],
                        r["fake_probability"], r["confidence"], r["uncertain"]])
    print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()
