# Amine Touihri

Final-year AI engineering student at ESPRIT, Tunisia, interested in building and evaluating applied machine learning systems. My work spans computer vision and explainable AI, audio classification and retrieval, and ML-backed applications.

I am looking for a PFE internship in that kind of work.

## Technical focus

- Computer vision and explainability: lesion-grouped evaluation, calibration, Grad-CAM, and mask overlap.
- Audio: a music / non-music gate, then spectral-fingerprint retrieval on live radio.
- Backend: Django REST and React around those models.

## Selected work

### [dermai-xai-tests](https://github.com/TouihriMedAmine/dermai-xai-tests)

PyTorch evaluation of the public EfficientNet-B0 checkpoint [`sgonzalez2000/dermai-efficientnet-b0`](https://huggingface.co/sgonzalez2000/dermai-efficientnet-b0), seven HAM10000 classes. The code is independent of the upstream repository.

The suite implements a lesion-grouped split, calibration (ECE), Grad-CAM, and mask overlap (IoU and a pointing-game hit). Offline tests are `pytest -m "not network"`. A full metric run is `scripts/evaluate.py` with a local HAM10000 copy. One RTX 4060 run on 22 Oct 2026 is tabulated in the repository README: balanced accuracy 0.949 on 1,002 images, ECE 0.017, IoU 0.467 ± 0.179, pointing-game hit rate 0.971.

That split is `StratifiedGroupKFold` on `lesion_id`, not the checkpoint authors' holdout, so the classification numbers can include images used to train the published weights. They are not an independent generalization estimate. Deletion and insertion scores from that day are withdrawn until the corrected multi-channel perturbation is re-run. HAM10000 is CC BY-NC-SA 4.0 and is not in this repository.

### [FairPlay.ai](https://github.com/TouihriMedAmine/FairPlay.ai)

Live radio monitor. `fetch.py` classifies each 5-second window with an MFCC model, then `audio_engine` matches a spectral fingerprint. Shazam is the fallback. The fingerprint tests run with `pytest` from `audio_engine/`. The classifier weights are in a private repository; the install steps are in the README.

The BiLSTM notebook is [`music-detection-with-rnn.ipynb`](https://github.com/TouihriMedAmine/FairPlay.ai/blob/main/MusicRecognition/Notebooks/music-detection-with-rnn.ipynb). Its second-folder log is 100 music mp3s and 100 non-music mp3s, each read for 5 seconds. A hit is an `argmax` that matches the folder name: 99/100 music and 100/100 non-music. The miss is `music_41.mp3`. That folder is a different Kaggle dataset from training. The notebook does not check filenames across the two sets, and it does not say how the 200 files were chosen. The printed 1.00 is a 20% split of the training set used as `validation_data`. The notebook uses 22,050 Hz. The live detector uses 16,000 Hz.

### MED-CASE (private)

Django REST and React trainer: a student works a clinical case, and a teacher publishes the grade. Keyword, summary, concept-map, and diagnosis scores are deterministic. A local Ollama model writes a separate review and does not replace the score. The repository and the classroom cases are private. I can walk through the code in an interview.

### [Water-Scarcity](https://github.com/TouihriMedAmine/Water-Scarcity)

Team project for the ESPRIT AI Project course, presented at the ESPRIT Tech Fair 2025. The app loads trained checkpoints. It does not include the training loops or a held-out metric table.

My modules:

- [Surface runoff](https://github.com/TouihriMedAmine/Water-Scarcity/tree/main/WaterScarcity/runoff): ConvLSTM checkpoint `best_qs_only.pth`.
- [Deforestation mask](https://github.com/TouihriMedAmine/Water-Scarcity/blob/main/WaterScarcity/Watershed/views.py): U-Net, 4 input channels, one mask channel, drawn on NDVI.
- [Drought inference](https://github.com/TouihriMedAmine/Water-Scarcity/tree/main/WaterScarcity/Drought): loads `drought-prediction.h5`.

Water level, irrigation, precipitation, and the chatbot were built by Maher Abderrahim and Houssem Bench. Clone instructions are in that README. There is no pinned `requirements.txt`.

## Technical stack

Python, PyTorch, TensorFlow/Keras, Django REST framework, React, and librosa.

## Contact

GitHub: https://github.com/TouihriMedAmine

Email: [amine.touihri@esprit.tn](mailto:amine.touihri@esprit.tn)
