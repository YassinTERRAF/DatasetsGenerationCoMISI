# CoMISI: Multimodal Speaker Identification Dataset Generation

This repository provides scripts to generate multimodal speaker identification datasets using the GRID and RAVDESS datasets. These datasets are essential for the study of speaker identification in diverse audio-visual conditions, including neutral and emotional states. The generated datasets will include audio noise, reverberation, and visual distortions.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Generating the GRID Dataset](#generating-the-grid-dataset)
- [Generating the RAVDESS Dataset](#generating-the-ravdess-dataset)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Overview

The provided scripts are part of the work published in:
> Yassin Terraf, Youssef Iraqi. "CoMISI: Multimodal Speaker Identification in Diverse Audio-Visual Conditions through Cross-Modal Interaction." Proceedings of the International Conference on Neural Information Processing (ICONIP), 2024. (Accepted for publication).

### GRID Dataset for Neutral Audio-Visual Data
- **Dataset**: The GRID dataset contains neutral speech and visual data.
- **Script**: `dataset_generation_grid.py`
- **Modifications**: Audio signals are distorted with noise and reverberation, while visual data is altered with noise, blur, and color shifts.

### RAVDESS Dataset for Emotional Audio-Visual Data
- **Dataset**: The RAVDESS dataset contains emotional speech and visual data, with multiple emotions such as calm, happy, sad, angry, and more.
- **Script**: `dataset_generation_ravdess.py`
- **Modifications**: Similar to the GRID dataset, audio signals in the RAVDESS dataset are distorted with noise and reverberation, and visual data is altered with noise, blur, and color shifts.

## Prerequisites

### Dependencies:
Make sure you have the following dependencies installed:

```bash
pip install numpy pandas librosa torchaudio tqdm opencv-python-headless pyroomacoustics keras_facenet speechbrain mtcnn
