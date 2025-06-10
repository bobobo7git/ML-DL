import torch
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2ForSpeechEmotionRecognition
import librosa
import numpy as np
import time

# Load model and feature extractor
model = Wav2Vec2ForSpeechEmotionRecognition.from_pretrained(
    "jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition2_data_rebalance",
    ignore_mismatched_sizes=True  # 내부적으로 구조 mismatch 허용
)
print(model.classifier)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition2_data_rebalance"
)

# Set model to evaluation mode
model.eval()

# Load and preprocess audio
input_path = "angry.mp3"
audio, sr = librosa.load(input_path, sr=None)
audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
audio = librosa.to_mono(audio)
audio = audio / np.max(np.abs(audio))  # normalize to [-1, 1]

inputs = feature_extractor(
    audio, sampling_rate=16000, return_tensors="pt"
)
id2 = model.config.id2label
print(id2)
# Inference
with torch.no_grad():
    start = time.time()
    logits = model(**inputs).logits
    end = time.time()
predicted_class_id = torch.argmax(logits, dim=-1).item()

id2label = {
    0: "기쁨",
    1: "당황",
    2: "분노",
    3: "불안",
    4: "슬픔",
    5: "중립"
}

print(f"Inference time: {end - start:.3f} seconds")
print(f'input file: {input_path}')
print(f"Detected emotion class: {id2label[predicted_class_id]}")
