# Fake Voice Detection

A machine learning system that classifies speech recordings as **real (human)** or **fake (AI-generated)**, with a web interface for analyzing single recordings and comparing two voices.

![Analyze a voice](docs/screenshot-analyze.png)

## Results

The model is evaluated with **5-fold grouped cross-validation**. Each fold holds out complete voices or sources, such as every clip from one synthetic voice, so the scores measure performance on **voices the model has never heard**.

| Metric | Baseline (24 features, neural network) | Final model |
|---|---|---|
| Accuracy | 88.3% | **95.1%** |
| ROC-AUC | — | **0.995** |
| Accuracy with added noise | — | 90.7% |
| Accuracy after MP3 compression | — | 89.6% |
| Accuracy on telephone-band audio | — | 92.9% |

## Getting started

Requires Python 3.10 or later. Tested on Python 3.13.

```bash
git clone https://github.com/h7al0x-ctrl/fake-voice-detection.git
cd fake-voice-detection

python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

python app.py
```

The web app opens at `http://127.0.0.1:5000`. A trained model is included in `model/`, so no training is needed to run it.

### Command-line prediction

```bash
python predict.py voice.wav
python predict.py path/to/folder --csv results.csv
```

### Training

The audio dataset is not included in the repository. To retrain, organize your data as follows:

```
dataset/
├── REAL/    # human speech
└── FAKE/    # AI-generated speech
```

Then run:

```bash
python train.py
```

The script reports cross-validated metrics and saves the model to `model/`. Supported formats: WAV, MP3, FLAC, OGG.

## How it works

1. **Preprocessing.** Audio is converted to 16 kHz mono, trimmed of leading and trailing silence, and peak-normalized. Resampling every clip to the same rate prevents the model from learning recording bandwidth instead of voice characteristics. In the original data, real and fake clips came at different sample rates.
2. **Feature extraction (228 features).** MFCCs with first- and second-order deltas, LFCCs, spectral centroid, bandwidth, roll-off, flatness and contrast, zero-crossing rate, energy, and pitch statistics. Each feature is summarized by its mean and standard deviation over time.
3. **Augmentation.** During training, each clip is also used with added noise, speed changes (0.9× and 1.1×), MP3 compression, and a telephone-band filter, so the model holds up on real-world recordings.
4. **Classifier.** A soft-voting ensemble of Logistic Regression, an RBF-kernel SVM, and Extra Trees, selected after comparing six models.
5. **Voice comparison.** Cosine similarity between normalized speaker embeddings built from MFCC and pitch statistics. Each voice is also classified as real or fake.

## Project structure

| File | Purpose |
|---|---|
| `features.py` | Audio loading, feature extraction, and augmentation, shared by training and inference |
| `train.py` | Training and grouped cross-validation |
| `detector.py` | Model loading, prediction, and voice comparison |
| `app.py` | Flask web server |
| `predict.py` | Command-line interface |
| `templates/index.html` | Web interface |
| `model/` | Trained model and evaluation metrics |

## Web interface

- Upload or drag and drop audio files
- Record directly from the microphone
- Waveform preview and playback
- Fake-probability meter, with a warning for uncertain results
- Side-by-side comparison of two voices
- Session history

![Compare two voices](docs/screenshot-compare.png)

## Limitations

- **Small dataset.** The model is trained on 91 real and 92 fake clips.
- **Single source of real speech.** All real recordings come from Mozilla Common Voice, so real speech from other sources, such as phone recordings, may occasionally be classified as fake.
- **Unseen generators.** Most errors occur on clips from a TTS generator that was entirely absent from training.
- **Voice similarity is approximate.** It is not a forensic speaker-verification tool.

## Future work

- Replace hand-crafted features with pretrained speech representations such as wav2vec 2.0 or WavLM.
- Expand the dataset with real speech from diverse sources and fake speech from more generators.
