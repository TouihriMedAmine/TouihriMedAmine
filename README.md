# Amine Touihri

Final-year Artificial Intelligence engineering student at ESPRIT, Tunisia, interested in building and evaluating applied machine learning systems. My work spans computer vision and explainable AI, audio classification and retrieval, and ML-backed applications.


## Technical Focus

- **Computer Vision & Explainable AI:** Lesion-grouped evaluation, model calibration, Grad-CAM, and segmentation-mask overlap metrics.
- **Real-Time Computer Vision Applications:** Webcam gaze estimation with per-user calibration, turned into a packaged Windows desktop tool.
- **Audio Classification & Retrieval:** Music detection using MFCC features, followed by spectral-fingerprint matching for live radio monitoring.
- **Backend & Full-Stack Engineering:** Django REST Framework and React applications integrating machine learning and AI-assisted workflows.

## Tech Stack

Technologies that appear in at least two projects.

**Languages**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black) ![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white) ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white) ![Dart](https://img.shields.io/badge/Dart-0175C2?style=for-the-badge&logo=dart&logoColor=white)

**Frontend**

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB) ![React Router](https://img.shields.io/badge/React_Router-CA4245?style=for-the-badge&logo=reactrouter&logoColor=white) ![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white) ![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white) ![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white) ![Flutter](https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white) ![Framer Motion](https://img.shields.io/badge/Framer_Motion-0055FF?style=for-the-badge&logo=framer&logoColor=white) ![Recharts](https://img.shields.io/badge/Recharts-22B5BF?style=for-the-badge&logo=recharts&logoColor=white) ![Axios](https://img.shields.io/badge/Axios-5A29E4?style=for-the-badge&logo=axios&logoColor=white)

**Backend**

![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white) ![Django REST](https://img.shields.io/badge/Django_REST-092E20?style=for-the-badge&logo=django&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) ![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white) ![JWT](https://img.shields.io/badge/JWT-000000?style=for-the-badge&logo=jsonwebtokens&logoColor=white) ![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)

**Machine learning and data**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white) ![Pillow](https://img.shields.io/badge/Pillow-8CACC8?style=for-the-badge&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white) ![Beautiful Soup](https://img.shields.io/badge/Beautiful_Soup-4B8BBE?style=for-the-badge&logoColor=white) ![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

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

## Contact

- **GitHub:** [TouihriMedAmine](https://github.com/TouihriMedAmine)
- **Email:** [amine.touihri@esprit.tn](mailto:amine.touihri@esprit.tn)
