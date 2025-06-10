# Load model directly
from transformers import AutoProcessor, Wav2Vec2ForSpeechClassification

processor = AutoProcessor.from_pretrained("jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition2_data_rebalance")
model = Wav2Vec2ForSpeechClassification.from_pretrained("jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition2_data_rebalance")