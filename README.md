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
The model was tested on 200 audio segments that it never seen before, the results were 199/200.
The checkpoint is private. 

### [Water-Scarcity](https://github.com/TouihriMedAmine/Water-Scarcity)

Team project for the ESPRIT AI Project course, presented at the ESPRIT Tech Fair 2025.
I added the runoff forecast and the deforestation / drought inference modules. Water level, irrigation, precipitation, and the chatbot were built by teammates.

## Contact

GitHub: https://github.com/TouihriMedAmine
Email: amine.touihri@esprit.tn
