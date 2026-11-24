"""
app.py — واجهة الويب لكشف الصوت المزيّف

التشغيل:
    python app.py
ثم يفتح المتصفح تلقائيًا على http://127.0.0.1:5000
"""
import os
import tempfile
import threading
import webbrowser

from flask import Flask, request, jsonify, render_template

from detector import Detector

ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg"}
MAX_MB = 25

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
detector = Detector()


def save_upload(field):
    """يحفظ الملف المرفوع في ملف مؤقت باسم فريد (عشان طلبين بنفس الوقت ما يتعارضون)."""
    f = request.files.get(field)
    if f is None or f.filename == "":
        raise ValueError("No file was selected")
    ext = os.path.splitext(f.filename)[1].lower() or ".wav"
    if ext not in ALLOWED_EXT:
        raise ValueError(f"File type {ext} is not supported. Use: {', '.join(sorted(ALLOWED_EXT))}")
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    f.save(path)
    return path


def cleanup(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def info():
    return jsonify(detector.metrics)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    path = None
    try:
        path = save_upload("file")
        result = detector.analyze(path)
        print(f"🔍 {request.files['file'].filename}: {result['result']} ({result['confidence']}%)")
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("analyze failed")
        return jsonify({"error": f"Could not read the audio file: {e}"}), 500
    finally:
        if path:
            cleanup(path)


@app.route("/api/compare", methods=["POST"])
def compare():
    p1 = p2 = None
    try:
        p1, p2 = save_upload("file1"), save_upload("file2")
        return jsonify(detector.compare(p1, p2))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("compare failed")
        return jsonify({"error": f"Could not read the audio files: {e}"}), 500
    finally:
        cleanup(*(p for p in (p1, p2) if p))


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File is larger than {MAX_MB} MB"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    url = f"http://127.0.0.1:{port}"
    if os.environ.get("WERKZEUG_RUN_MAIN") is None:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"🚀 التطبيق شغّال على {url}")
    app.run(host="127.0.0.1", port=port, debug=False)
