import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import kagglehub
import shutil

# Загрузка датасета
path = kagglehub.dataset_download("ejlok1/surrey-audiovisual-expressed-emotion-savee")
data_folder = path + '/ALL'
print("Path to dataset files:", data_folder)

# Эмоции и их коды
emotion_map = {
    "a": "anger",
    "d": "disgust",
    "f": "fear",
    "h": "happiness",
    "sa": "sadness",
    "su": "surprise",
    "n": "neutral"
}

# Функция для аугментации данных
def augment_audio(audio, sr):
    noise = np.random.normal(0, 0.005, audio.shape)
    audio_noisy = audio + noise
    audio_shifted = np.roll(audio, shift=np.random.randint(sr // 10))
    audio_stretched = librosa.effects.time_stretch(audio, rate=1.1)
    audio_pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    return [audio, audio_noisy, audio_shifted, audio_stretched, audio_pitched]

# Функция для извлечения MEL-спектрограмм
def extract_features(file_path, max_pad_len=128):
    audio, sr = librosa.load(file_path, sr=22050)
    audio = librosa.effects.trim(audio)[0]  # Удаление тишины
    audio = librosa.util.normalize(audio)  # Нормализация громкости
    augmented_audios = augment_audio(audio, sr)
    features = []
    for aug_audio in augmented_audios:
        mel_spec = librosa.feature.melspectrogram(y=aug_audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec_db.shape[1] < max_pad_len:
            pad_width = max_pad_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_pad_len]
        features.append(mel_spec_db)
    return features

# Загрузка данных
X, y = [], []
for file_name in os.listdir(data_folder):
    if file_name.endswith(".wav"):
        parts = file_name.split("-")
        emotion_code = parts[2]
        if emotion_code in emotion_map:
            file_path = os.path.join(data_folder, file_name)
            feature_set = extract_features(file_path)
            X.extend(feature_set)
            y.extend([emotion_map[emotion_code]] * len(feature_set))

# Преобразование в numpy массивы
X = np.array(X)
y = np.array(y)

# Кодировка меток эмоций
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"Размер обучающего набора: {X_train.shape}")
print(f"Размер тестового набора: {X_test.shape}")

# Улучшенная модель
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(emotion_map), activation='softmax')
])

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Параметры модели
epochs = 50
batch_size = 32

# Обучение модели
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                    callbacks=[early_stopping, lr_scheduler])

# Сохранение модели
model.save(f"models/SAVEE/emotion_recognition_model_{epochs}_epochs_SAVEE.keras")
print("Модель обучена и сохранена.")

# Построение и сохранение графиков
plt.figure(figsize=(12, 5))

# График потерь
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.title('График потерь')

# График точности
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.title('График точности')

# Сохранение графика в файл
plt.savefig(f"plots/SAVEE/training_plots_{epochs}_epochs_SAVEE.png")
print("Графики сохранены в training_plots.png")
