import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import sounddevice as sd
import scipy.io.wavfile as wavfile
import os
import tensorflow as tf

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load trained model
model = load_model(r"C:\Users\mishr\OneDrive\Desktop\parkinsons_voice_detection\parkinsons_voice_model.h5")

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=2.5, offset=0.6)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

def predict(file_path):
    features = extract_features(file_path)
    if features is not None:
        features = np.expand_dims(features, axis=0)  # Reshape for model
        prediction = model.predict(features)
        result = np.argmax(prediction, axis=1)[0]
        return "Parkinson's Disease" if result == 1 else "Healthy Control"
    return None

# Streamlit UI
st.title("Parkinson's Disease Voice Detection")
option = st.selectbox("Choose an option:", ["Upload Recorded Audio", "Real-Time Recording"])

if option == "Upload Recorded Audio":
    uploaded_file = st.file_uploader("Choose a .wav file", type="wav")
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Predict"):
            result = predict(file_path)
            st.success(f"Prediction: {result}")
            os.remove(file_path)  # Clean up the uploaded file

elif option == "Real-Time Recording":
    if st.button("Record"):
        duration = 3  # seconds
        fs = 22050  # Sample rate
        st.write("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        wavfile.write("realtime_input.wav", fs, recording)
        st.write("Recording complete.")
        
        result = predict("realtime_input.wav")
        st.success(f"Prediction: {result}")
        os.remove("realtime_input.wav")  # Clean up the recorded file
