import telebot
import os
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
from config import TOKEN

bot = telebot.TeleBot(TOKEN)

# Загрузка модели
MODEL_PATH = "../models/RAVDESS/emotion_recognition_model_1000_epochs_RAVDESS.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Эмоции
# emotion_labels = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# Приветственное сообщение
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправьте мне голосовое сообщение, и я определю эмоцию.")
    model.summary()


# Обработка голосового сообщения
@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    number = len(os.listdir())
    with open(f"temp_audio_{number}.ogg", 'wb') as new_file:
        new_file.write(downloaded_file)

    # Конвертация в WAV
    os.system(f"ffmpeg -i temp_audio_{number}.ogg temp_audio_{number}.wav -y")

    # Извлечение признаков
    def extract_features(file_path, max_pad_len=128):
        audio, sr = librosa.load(file_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec_db.shape[1] < max_pad_len:
            pad_width = max_pad_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_pad_len]
        return mel_spec_db

    features = extract_features(f"temp_audio_{number}.wav")
    features = np.expand_dims(features, axis=[0, -1])
    prediction = model.predict(features)

    # emotion = emotion_labels[np.argmax(prediction)]

    emotion_code = str(np.argmax(prediction) + 1).zfill(2)
    emotion = emotion_labels.get(emotion_code, "неизвестно")

    bot.reply_to(message, f"Эмоция: {emotion}")
    print(f"Эмоция: {emotion}")

    #os.remove("temp_audio.ogg")
    #os.remove("temp_audio.wav")


# Обработка всех остальных сообщений
@bot.message_handler(func=lambda message: True)
def handle_other_messages(message):
    bot.reply_to(message, "Пожалуйста, отправьте голосовое сообщение.")


bot.polling()
