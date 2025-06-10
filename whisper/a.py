# Whisper-based Korean Speech Emotion Recognition (Official Whisper-compatible Structure)

from datasets import Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import torchaudio
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------------------
# 1. Config
# ----------------------------------
MODEL_ID = "openai/whisper-large-v3"
CSV_PATH = "/content/labels.csv"  # CSV 파일 (wav_id, final_emotion)
AUDIO_DIR = "/content/audio_files"  # WAV 파일 디렉토리
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
LABEL2ID = {emo: i for i, emo in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: emo for emo, i in LABEL2ID.items()}
TARGET_SR = 16000
MAX_DURATION = 30.0  # 초
MAX_SAMPLES_PER_CLASS = 1000  # 클래스당 최대 샘플 수 (undersampling)

# ----------------------------------
# 2. Dataset loading
# ----------------------------------
def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)
    waveform = waveform.mean(dim=0)  # mono
    waveform = waveform / waveform.abs().max()  # normalize
    max_len = int(TARGET_SR * MAX_DURATION)
    if waveform.shape[0] > max_len:
        waveform = waveform[:max_len]
    else:
        padding = torch.zeros(max_len - waveform.shape[0])
        waveform = torch.cat((waveform, padding))
    return waveform

def get_dataset_dict():
    df = pd.read_csv(CSV_PATH, index_col=0)
    df = df[df['final_emotion'].isin(EMOTION_LABELS)]
    df_balanced = pd.concat([
        df[df['final_emotion'] == emo].sample(n=MAX_SAMPLES_PER_CLASS, random_state=42, replace=False)
        if len(df[df['final_emotion'] == emo]) > MAX_SAMPLES_PER_CLASS else df[df['final_emotion'] == emo]
        for emo in EMOTION_LABELS
    ])
    return df_balanced.reset_index()

# ----------------------------------
# 3. Feature extraction & dataset creation (streamed)
# ----------------------------------
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

def build_dataset(df):
    input_features, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = os.path.join(AUDIO_DIR, row['wav_id'] + ".wav")
        if not os.path.exists(path):
            continue
        waveform = load_audio(path)
        inputs = extractor(waveform.numpy(), sampling_rate=TARGET_SR, return_tensors="np")
        input_features.append(inputs['input_features'][0].astype(np.float32))
        labels.append(LABEL2ID[row['final_emotion']])
    return Dataset.from_dict({
        'input_features': input_features,
        'label': labels
    })

raw_df = get_dataset_dict()
dataset = build_dataset(raw_df)

# ----------------------------------
# 4. Model init
# ----------------------------------
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(EMOTION_LABELS),
    label2id=LABEL2ID,
    id2label=ID2LABEL,
    ignore_mismatched_sizes=True
)

# ----------------------------------
# 5. Training
# ----------------------------------
training_args = TrainingArguments(
    output_dir="./whisper-korean-emotion",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    gradient_accumulation_steps=4,
    fp16=True,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset.train_test_split(test_size=0.1)["test"]
)

trainer.train()
