# Audio-to-Affect Prototype

## Overview
This is an **end-to-end prototype** that converts raw audio into **latent affect trajectories** and computes **stability metrics**. The system uses a pretrained `wav2vec2` model and a BiLSTM temporal head to generate Arousal, Valence, Dominance (A/V/D) trajectories. Outputs include CSV, JSON, and visual plots.

**Key Features:**
- Raw audio → embeddings → latent A/V/D trajectory
- Stability metrics: variance, mean absolute delta, baseline drift
- EMA smoothing applied to latent trajectory
- Outputs: `trajectory.csv`, `summary.json`, time-series & radar plots
- Lightweight, reproducible, and extendable
- Non-clinical, non-diagnostic prototype

## Requirements
- Python 3.10+
- PyTorch
- transformers
- librosa
- numpy
- pandas
- matplotlib

(Full list in `requirements.txt`)

## Usage
1. Place a WAV file in the folder (or use existing sample audio).
2. Run the pipeline:
```bash
latent_head_test.py
