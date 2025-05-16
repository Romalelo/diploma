import tensorflow as tf
import numpy as np
import librosa


# Функция для загрузки аудиофайлов и конвертации в MEL-спектрограммы
def extract_features(file_path, max_pad_len=128):
    audio, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Нормализация и паддинг
    if mel_spec_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_pad_len]

    return mel_spec_db


# Функция для предсказания эмоции
def predict_emotion(model_path, file_path):
    model = tf.keras.models.load_model(model_path)
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)
    emotion_classes = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    return emotion_classes[predicted_label]


if __name__ == "__main__":
    model_path = 'models/SAVEE/emotion_recognition_model_500_epochs_SAVEE.keras'
    #audio_file = 'Грусть.wav'
    #audio_file = 'Злость.wav'
    #audio_file = 'Нейтральное.wav'
    audio_file = 'Радость.wav'
    emotion = predict_emotion(model_path, audio_file)
    print(f"Предсказанная эмоция: {emotion}")
