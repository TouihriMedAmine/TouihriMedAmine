# Amine Touihri

Final-year Artificial Intelligence engineering student at ESPRIT, Tunisia, interested in building and evaluating applied machine learning systems. My work spans computer vision and explainable AI, audio classification and retrieval, and ML-backed applications.


## Technical Focus

- **Computer Vision & Explainable AI:** Lesion-grouped evaluation, model calibration, Grad-CAM, and segmentation-mask overlap metrics.
- **Real-Time Computer Vision Applications:** Webcam gaze estimation with per-user calibration, turned into a packaged Windows desktop tool.
- **Audio Classification & Retrieval:** Music detection using MFCC features, followed by spectral-fingerprint matching for live radio monitoring.
- **Backend & Full-Stack Engineering:** Django REST Framework and React applications integrating machine learning and AI-assisted workflows.

## Selected Work

### [dermai-xai-tests](https://github.com/TouihriMedAmine/dermai-xai-tests)

Independent PyTorch evaluation suite for the public [EfficientNet-B0 checkpoint](https://huggingface.co/sgonzalez2000/dermai-efficientnet-b0), fine-tuned for seven HAM10000 skin-lesion classes. The evaluation code is independent of the upstream repository.

The suite implements:

- Lesion-grouped data splitting using `StratifiedGroupKFold`.
- Classification calibration measured with Expected Calibration Error (ECE).
- Grad-CAM visual explanations.
- Explanation localization metrics, including Intersection over Union (IoU) and pointing-game hit rate.

Offline tests can be run with:

```bash
pytest -m "not network"
```

A full evaluation requires a local copy of HAM10000 and can be launched with:

```bash
python scripts/evaluate.py
```

One RTX 4060 run, performed on 22 October 2026, produced the following results on 1,002 images:

| Metric | Result |
|---|---:|
| Balanced accuracy | 0.949 |
| Expected Calibration Error | 0.017 |
| Grad-CAM mask IoU | 0.467 ± 0.179 |
| Pointing-game hit rate | 0.971 |

**Evaluation limitations:** The lesion-grouped split is not the original checkpoint authors' holdout set. Because the checkpoint was already trained, the evaluation images may overlap with images used to train the published weights. These classification metrics are therefore not an independent estimate of generalization to unseen lesions.

Deletion and insertion scores from the same run have been withdrawn pending a corrected multi-channel perturbation implementation and a rerun.

HAM10000 is licensed under CC BY-NC-SA 4.0 and is not redistributed in this repository.

### [FairPlay.ai](https://github.com/TouihriMedAmine/FairPlay.ai)

A live radio-monitoring system that detects music and identifies tracks using a combination of audio classification and fingerprint-based retrieval.

**How it works**

1. **Music detection:** `fetch.py` processes five-second audio windows using an MFCC-based classifier.
2. **Track identification:** The `audio_engine` module searches a local spectral-fingerprint index.
3. **Fallback identification:** Shazam is used when local fingerprint matching does not identify the track.

The audio datasets were manually collected and prepared for the project, then uploaded to Kaggle for use in model development and evaluation.

**Classifier evaluation**

The BiLSTM implementation is documented in the [music-detection-with-rnn.ipynb notebook](https://github.com/TouihriMedAmine/FairPlay.ai/blob/main/MusicRecognition/Notebooks/music-detection-with-rnn.ipynb).

A second-folder evaluation used 100 music MP3 files and 100 non-music MP3 files, with each file evaluated on a five-second segment.

| Class | Correct predictions | Total |
|---|---:|---:|
| Music | 99 | 100 |
| Non-music | 100 | 100 |

The single music-classification miss was `music_41.mp3`, whose evaluated segment was empty.

The evaluation set came from a different Kaggle dataset than the training data. However, filenames were not checked across the two datasets, and the selection procedure for the 200 files was not documented. These results should be interpreted as a small, dataset-specific evaluation rather than a robust estimate of generalization.

The notebook uses a sampling rate of 22,050 Hz, while the live detector uses 16,000 Hz. Its printed validation accuracy of 1.00 comes from a 20% split of the training data and should not be confused with the separate 200-file evaluation.

**Live-system performance**

During my live testing, the system has not missed a song so far. This is an informal observation of real-world use rather than a controlled accuracy benchmark.

The fingerprint tests can be run from `audio_engine/` using `pytest`. Classifier weights are maintained in a private repository; installation instructions are available in the project README.

### [GazeSwitch](https://github.com/TouihriMedAmine/GazeSwitch)

A Windows tray app that uses an ordinary webcam to give keyboard focus to the window on whichever of three monitors I am looking at.

**How it works**

1. **Tracking:** MediaPipe Face Landmarker (478 landmarks) runs at about 38 fps; each frame yields head yaw and iris position relative to the eye corners.
2. **Per-user gaze model:** A guided calibration visits the left, center, and right monitors while the user keeps their eyes on each one and moves their head. A regularised least-squares fit then learns how much the eyes compensate for head rotation, so a head turned left with eyes on the center screen still counts as center. On my setup the fitted iris weight is about ten times the head-yaw weight, the opposite of the fixed formula it replaced.
3. **Selection and focus:** A time-windowed vote with dwell and hysteresis picks the monitor; Win32 calls then focus the front-most window on it, working around the Windows foreground lock.

The app ships as a portable PyInstaller build with a tray menu, global shortcuts, and live camera switching. 107 offline tests run with `pytest`.

**Evaluation**

`scripts/evaluate_gaze.py` shows a cue on a random monitor (never the same one twice in a row) and records which monitor the app selects 2.5 s later. One run on 27 October 2026:

| Looked at \ Detected | Left | Center | Right | Unknown |
|---|---:|---:|---:|---:|
| Left | 10 | 0 | 0 | 0 |
| Center | 0 | 10 | 0 | 0 |
| Right | 0 | 0 | 10 | 0 |

| Metric | Result |
|---|---:|
| Monitor accuracy | 30 / 30 |
| Cue → monitor selected (median, p90) | 1.51 s, 1.75 s |
| Selection → window focused (median, p90; 305 switches in daily use) | 208 ms, 1097 ms |

**Evaluation limitations:** One user, one desk, and one camera, tested two minutes after calibration. Cues sat in the middle of each monitor, so screen edges and posture drift over a day are not covered, and 30 cues is a small sample. The cue-to-selection time includes my own reaction and head movement. The slow tail of the focus latency is mostly deliberate: focus is not taken while I am typing. The tracker works at monitor level, not as a pixel-level gaze estimate.

## Technical Stack

**Languages:** Python

**Machine Learning & Deep Learning:** PyTorch, TensorFlow/Keras

**Computer Vision & Audio:** OpenCV, MediaPipe, Grad-CAM, U-Net, ConvLSTM, librosa, MFCCs, spectral fingerprinting

**Backend & Frontend:** Django REST Framework, React

**Testing & Deployment:** pytest, local inference with Ollama, PyInstaller

## Contact

- **GitHub:** [TouihriMedAmine](https://github.com/TouihriMedAmine)
- **Email:** [amine.touihri@esprit.tn](mailto:amine.touihri@esprit.tn)
