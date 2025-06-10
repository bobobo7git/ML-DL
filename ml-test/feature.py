import opensmile
import pandas as pd
import os
from joblib import Parallel, delayed
import time

def extract(filepath):
    features = smile.process_file(filepath)
    features["wav_id"] = os.path.splitext(os.path.basename(filepath))[0]
    return features

start_time = time.time()

# 피처 추출기 설정
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals
)

# 분석할 음성 파일 리스트
# audio_dirs = ["samples/4th", "samples/5th_1st", "samples/5th_2nd"]
audio_dirs = ['samples']
file_list = []
for dir in audio_dirs:
    file_list += [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".wav") or f.endswith(".mp3")]

# 전체 결과 저장용
all_features = []
data_id = "wav_id"


results = Parallel(n_jobs=8, verbose=10)(delayed(extract)(f) for f in file_list)

df = pd.concat(results, axis=0).set_index(data_id)

# # 보기 좋게 출력 (선택)
print(df.head())

# # CSV로 저장 (선택)
df.to_csv("test.csv")

# print(time.time() - start_time)
