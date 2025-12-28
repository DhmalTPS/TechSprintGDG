import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np
import matplotlib.pyplot as plt






# ---------- 1. LOAD AUDIO ----------
wav_path = "sample.wav"
#waveform, sr = torchaudio.load(wav_path)
import librosa
waveform, sr = librosa.load(wav_path, sr=16000)
waveform = torch.tensor(waveform).unsqueeze(0)

if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)
    sr = 16000

speech = waveform.squeeze().numpy()






# ---------- 2. LOAD WAV2VEC ----------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec.eval()

# IMPORTANT: freeze wav2vec
for p in wav2vec.parameters():
    p.requires_grad = False

inputs = processor(
    speech,
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = wav2vec(**inputs)

embeddings = outputs.last_hidden_state  # [1, T, 768]

print("Embedding shape:", embeddings.shape)






# ---------- 3. TEMPORAL HEAD ----------
class TemporalAffectHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(128 * 2, 3)  # A, V, D

    def forward(self, x):
        h, _ = self.lstm(x)
        z = self.fc(h)
        return z

head = TemporalAffectHead()
head.eval()






# ---------- 4. FORWARD PASS ----------
with torch.no_grad():
    Z = head(embeddings)   # [1, T, 3]

Z = Z.squeeze(0).numpy()   # [T, 3]

def ema_smooth(z, alpha=0.2):
    z_smooth = z.copy()
    for t in range(1, len(z)):
        z_smooth[t] = alpha * z[t] + (1 - alpha) * z_smooth[t - 1]
    return z_smooth

Z = ema_smooth(Z, alpha=0.2)
latent_z = Z

print("Latent Z shape:", Z.shape)






# ---------- 5. PLOT (SANITY CHECK) ----------
plt.figure(figsize=(10, 4))
plt.plot(Z[:, 0], label="Arousal")
plt.plot(Z[:, 1], label="Valence")
plt.plot(Z[:, 2], label="Dominance")
plt.legend()
plt.title("Latent Affect Trajectories (UNTRAINED)")
plt.tight_layout()
plt.show()







from stability_metrics import compute_stability

metrics = compute_stability(Z)

print("\n--- Stability Metrics ---")
for k, v in metrics.items():
    print(k, v)

stability_score = metrics["stability_score"]










summary = {
    "mean": Z.mean(axis=0).tolist(),
    "variance": Z.var(axis=0).tolist(),
    "mean_abs_delta": np.mean(np.abs(np.diff(Z, axis=0)), axis=0).tolist(),
    "drift": float(np.linalg.norm(Z[-1] - Z[0])),
    "stability_score": float(stability_score)
}

import json
with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Saved summary.json")








radar_values = {
    "arousal": float(Z[:, 0].std()),
    "valence": float(Z[:, 1].std()),
    "dominance": float(Z[:, 2].std())
}

print("Radar values:", radar_values)










labels = list(radar_values.keys())
values = list(radar_values.values())

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.3)

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Latent Affect Stability Profile")

plt.show()









import pandas as pd

df = pd.DataFrame(
    Z,
    columns=["arousal", "valence", "dominance"]
)

df.to_csv("trajectory.csv", index=False)

print("Saved trajectory.csv")