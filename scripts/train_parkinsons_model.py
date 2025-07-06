import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib
from collections import Counter

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Constants
DATA_PATH = r'C:\Users\mishr\OneDrive\Desktop\parkinsons_voice_detection\data\voice'
MAX_LEN = 173  # Based on typical MFCC time steps at 2.5s
N_MFCC = 40

def extract_mfcc_sequence(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=2.5, offset=0.6)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        return mfcc.T  # shape = (time_steps, n_mfcc)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_data():
    features = []
    labels = []

    for label in ['HC_AH', 'PD_AH']:
        folder = os.path.join(DATA_PATH, label)
        for file_name in os.listdir(folder):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder, file_name)
                mfcc_seq = extract_mfcc_sequence(file_path)
                if mfcc_seq is not None:
                    features.append(mfcc_seq)
                    labels.append(label)
    return features, labels

# Load data
X_seq, y = load_data()

# Pad sequences
X_padded = pad_sequences(X_seq, maxlen=MAX_LEN, dtype='float32', padding='post', truncating='post')

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save the label encoder
joblib.dump(le, "label_encoder.pkl")

# Stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X_padded, y_encoded):
    X_train, X_test = X_padded[train_index], X_padded[test_index]
    y_train, y_test = y_categorical[train_index], y_categorical[test_index]

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights_dict = dict(enumerate(class_weights))

# Build LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(MAX_LEN, N_MFCC)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("parkinsons_voice_model_best.h5", monitor='val_loss', save_best_only=True)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save("parkinsons_voice_model.h5")
print("Model saved successfully.")

# Plotting
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy over Epochs")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss over Epochs")
plt.legend()
plt.show()

# Test predictions check
y_pred = model.predict(X_test)
print("Sample Predictions (class index):", np.argmax(y_pred[:10], axis=1))
print("True Labels (class index):", np.argmax(y_test[:10], axis=1))

print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

print("Label counts:", Counter(y))

print("TensorFlow version:", tf.__version__)

print("Mean prediction probabilities:", np.mean(y_pred, axis=0))
