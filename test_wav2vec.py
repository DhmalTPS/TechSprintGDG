import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
#import torchaudio

# 1. Load audio
# waveform, sr = torchaudio.load(wav_path)
# if waveform.shape[0] > 1:  # convert to mono if stereo
#     waveform = waveform.mean(dim=0, keepdim=True)
# if sr != 16000:            # resample if needed
#     resampler = torchaudio.transforms.Resample(sr, 16000)
#     waveform = resampler(waveform)
#     sr = 16000
# speech = waveform.squeeze().numpy()
wav_path = "sample.wav"
speech, sr = librosa.load(wav_path, sr=16000)
speech = librosa.to_mono(speech)
print("Audio shape:", speech.shape)
print("Sample rate:", sr)

# 2. Load pretrained wav2vec2
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
)
model.eval()

# 3. Preprocess audio
inputs = processor(
    speech,
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)

# 4. Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# 5. Extract embeddings
embeddings = outputs.last_hidden_state
print("Embedding tensor shape:", embeddings.shape)
