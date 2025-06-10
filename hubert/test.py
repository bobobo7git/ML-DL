import torch
import librosa
from transformers import AutoFeatureExtractor, AutoConfig
import whisper
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch import nn
from transformers import HubertForSequenceClassification

class MyLitModel(pl.LightningModule):
    def __init__(self, audio_model_name, num_label2s, n_layers=1, projector=True, classifier=True, dropout=0.07, lr_decay=1):
        super(MyLitModel, self).__init__()
        self.config = AutoConfig.from_pretrained(audio_model_name)
        self.config.output_hidden_states = True
        self.audio_model = HubertForSequenceClassification.from_pretrained(audio_model_name, config=self.config)
        self.label2_classifier = nn.Linear(self.audio_model.config.hidden_size, num_label2s)
        self.intensity_regressor = nn.Linear(self.audio_model.config.hidden_size, 1)

    def forward(self, audio_values, audio_attn_mask=None):
        outputs = self.audio_model(input_values=audio_values, attention_mask=audio_attn_mask)
        label2_logits = self.label2_classifier(outputs.hidden_states[-1][:, 0, :])
        intensity_preds = self.intensity_regressor(outputs.hidden_states[-1][:, 0, :]).squeeze(-1)
        return label2_logits, intensity_preds

# 모델 관련 설정
audio_model_name = "team-lucid/hubert-base-korean"
NUM_LABELS = 7
SAMPLING_RATE = 16000

# Hubert 모델 로드
pretrained_model_path = "model.ckpt" # 모델 체크포인트
hubert_model = MyLitModel.load_from_checkpoint(
    pretrained_model_path,
    audio_model_name=audio_model_name,
    num_label2s=NUM_LABELS,
)
hubert_model.eval()
hubert_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Feature extractor 로드
feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)

# 음성 파일 처리
audio_path = "sad_03_8.wav"  # 처리할 음성 파일 경로
audio_np, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
inputs = feature_extractor(raw_speech=audio_np, return_tensors="pt", sampling_rate=SAMPLING_RATE)
audio_values = inputs["input_values"].to(hubert_model.device)
audio_attn_mask = inputs.get("attention_mask", None)
if audio_attn_mask is not None:
    audio_attn_mask = audio_attn_mask.to(hubert_model.device)

# 감정 분석
with torch.no_grad():
    if audio_attn_mask is None:
        label2_logits, intensity_preds = hubert_model(audio_values)
    else:
        label2_logits, intensity_preds = hubert_model(audio_values, audio_attn_mask)

emotion_label = torch.argmax(label2_logits, dim=-1).item()
emotion_intensity = intensity_preds.item()

print(f"Emotion Label: {emotion_label}, Emotion Intensity: {emotion_intensity}")
