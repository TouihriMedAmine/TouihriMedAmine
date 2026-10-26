# Amine Touihri

Final-year Artificial Intelligence engineering student at ESPRIT, Tunisia.
I am looking for a PFE internship in applied machine learning, computer vision, or backend engineering.

## What I can show

- PyTorch evaluation of a vision model: grouped splits, calibration, Grad-CAM, and mask overlap.
- A live radio monitor: an MFCC music gate, then a spectral-fingerprint index.
- Django REST and React applications, including a private clinical-case trainer for medical students.
- Coursework in satellite and environmental models (U-Net, ConvLSTM), done with a team. My part was surface runoff and deforestation / drought inference.

## Selected work

### [dermai-xai-tests](https://github.com/TouihriMedAmine/dermai-xai-tests)

Independent tests and evaluation scripts for a public EfficientNet-B0 checkpoint fine-tuned on HAM10000.
The reported classification metrics use a lesion-grouped fold, which can overlap the authors' training images. That limitation is written in the repository README.
Deletion and insertion scores from 22 Oct 2026 are withdrawn until the corrected multi-channel perturbation is re-run.
HAM10000 is CC BY-NC-SA 4.0 and is not redistributed here.

### [FairPlay.ai](https://github.com/TouihriMedAmine/FairPlay.ai)

Live monitor for radio airplay. A 5-second MFCC model decides whether a window is music, then a local fingerprint index names the song, with Shazam as the fallback.
On a second Kaggle folder, 100 music clips and 100 non-music clips, the notebook log shows 199 correct. The line that prints 1.00 is a 20% split of the training set used as validation during training.
The checkpoint is private. That notebook run uses 22,050 Hz; the live detector uses 16,000 Hz.

### MED-CASE (private)

Django and React platform where a student works a clinical case and a teacher publishes the grade.
Keyword, summary, concept-map, and diagnosis scores are deterministic. A local Ollama model writes a separate reasoning review and does not replace that score.
I can walk through the code in an interview. The classroom cases stay private.

### [Water-Scarcity](https://github.com/TouihriMedAmine/Water-Scarcity)

Team project for the ESPRIT AI Project course, presented at the ESPRIT Tech Fair 2025.
I added the runoff forecast and the deforestation / drought inference modules. Water level, irrigation, precipitation, and the chatbot were built by teammates.

## Contact

GitHub: https://github.com/TouihriMedAmine
Email: amine.touihri@esprit.tn
