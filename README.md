#### HEAD
"# SafeVisionAI-ML" 
=======
# SafeVisionAI-ML
SafeVisionAi 
>>>>>>> aade0c789cd8548b190c803fbc6fa211ba51ce09
# SafeVisionAI-ML  

AI-powered Harassment & Violence Detection (Video + Audio Fusion)

SafeVisionAI is a research prototype that uses **computer vision** and **audio analysis** to detect violent or abnormal activity from CCTV and environmental recordings.  
The system processes **video frames (fights, anomalies)** and **audio events (screams, gunshots, glass breaking)**, then combines them into a **risk score** for real-time alerts.  

---

## Features
- **Video branch**: Detects violent events using CNN/3D-CNN on surveillance clips.  
- **Audio branch**: Classifies critical sounds like screams, gunshots, or glass breaks using spectrogram-based CNNs.  
- **Fusion model**: Combines predictions from video and audio into a unified risk score.  
- **Extensible**: Built with modular `src/` code and `notebooks/` for experiments.  

---

## Project Structure
SafeVisionAI-ML/
│
├── data/ # datasets (not included in repo; see below)
│ ├── rwf2000/ # CCTV fight vs non-fight
│ ├── mivia_audio/ # screams, gunshots, glass breaking
│ └── xd_violence/ # multimodal dataset (video + audio)
│
├── notebooks/ # Jupyter experiments
│ ├── video_model.ipynb
│ ├── audio_model.ipynb
│ └── fusion_model.ipynb
│
├── src/ # source code
│ ├── data_loader.py # dataset loaders
│ ├── video_model.py # CNN/3D-CNN for video
│ ├── audio_model.py # CNN for audio
│ ├── fusion_model.py # late/early fusion logic
│ └── train.py # main training script
│
├── requirements.txt # dependencies
├── README.md # this file
└── .gitignore

---

## Datasets
This project uses **publicly available research datasets** (not included in repo):

1. **Video violence detection**  
   - [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database) – ~2000 CCTV fight/non-fight clips.  
   - [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) – 1,900 surveillance anomaly videos.  
   - [XD-Violence](https://roc-ng.github.io/XD-Violence.html) – 217 hours, audio+video, weakly labeled violence dataset.  

2. **Audio events**  
   - [MIVIA Audio Events](http://mivia.unisa.it/datasets/audio-analysis/mivia-audio-events/) – screams, gunshots, glass breaking.  
   - [AudioSet](https://research.google.com/audioset/) (subset: screaming, gunshots).  

Download links are subject to license/academic request. Place datasets inside `data/` following the structure above.

---

## ⚙️ Installation
Clone the repo:
```bash
git clone https://github.com/Hi-Chaudhary/SafeVisionAI-ML.git
cd SafeVisionAI-ML