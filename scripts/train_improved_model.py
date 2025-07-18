import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from collections import Counter

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dropout, Dense, BatchNormalization, 
                                   Conv1D, MaxPooling1D, GlobalAveragePooling1D, 
                                   Bidirectional, Input, Concatenate, Attention)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Constants
DATA_PATH = r'C:\Users\mishr\OneDrive\Desktop\Prediction-of-Parkinson-s-Disease\data\voice'
MAX_LEN = 200  # Increased for better temporal resolution
N_MFCC = 40
N_CHROMA = 12
N_MEL = 128
HOP_LENGTH = 512

def extract_enhanced_features(file_path, augment=False):
    """Extract multiple audio features including MFCC, Chroma, Mel-spectrogram, and statistical features"""
    try:
        # Load audio with different parameters for robustness
        audio, sr = librosa.load(file_path, duration=3.0, offset=0.5)
        
        # Data augmentation if requested
        if augment:
            # Random noise
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.005, audio.shape)
                audio = audio + noise
            
            # Time stretching
            if np.random.random() > 0.5:
                stretch_rate = np.random.uniform(0.8, 1.2)
                audio = librosa.effects.time_stretch(audio, rate=stretch_rate)
            
            # Pitch shifting
            if np.random.random() > 0.5:
                n_steps = np.random.uniform(-2, 2)
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        
        # Feature extraction
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP_LENGTH)
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MEL, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=HOP_LENGTH)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
        
        # Combine all features
        features = np.vstack([
            mfcc,
            mfcc_delta,
            mfcc_delta2,
            chroma,
            mel_spec_db[:20],  # Use first 20 mel bands
            spectral_centroid,
            spectral_rolloff,
            spectral_bandwidth,
            zero_crossing_rate
        ])
        
        return features.T  # shape = (time_steps, n_features)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_enhanced_data(use_augmentation=True):
    """Load data with enhanced features and optional augmentation"""
    features = []
    labels = []

    for label in ['HC_AH', 'PD_AH']:
        folder = os.path.join(DATA_PATH, label)
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        
        for file_name in files:
            file_path = os.path.join(folder, file_name)
            
            # Original features
            feature_seq = extract_enhanced_features(file_path, augment=False)
            if feature_seq is not None:
                features.append(feature_seq)
                labels.append(label)
            
            # Augmented features (only for training data)
            if use_augmentation:
                for _ in range(2):  # Create 2 augmented versions per original
                    aug_feature_seq = extract_enhanced_features(file_path, augment=True)
                    if aug_feature_seq is not None:
                        features.append(aug_feature_seq)
                        labels.append(label)
    
    return features, labels

def create_advanced_model(input_shape):
    """Create an advanced model with multiple architectures combined"""
    
    input_layer = Input(shape=input_shape)
    
    # CNN branch for local patterns
    cnn_branch = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = Conv1D(64, 3, activation='relu', padding='same')(cnn_branch)
    cnn_branch = MaxPooling1D(2)(cnn_branch)
    cnn_branch = Dropout(0.3)(cnn_branch)
    
    cnn_branch = Conv1D(128, 3, activation='relu', padding='same')(cnn_branch)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = Conv1D(128, 3, activation='relu', padding='same')(cnn_branch)
    cnn_branch = MaxPooling1D(2)(cnn_branch)
    cnn_branch = Dropout(0.3)(cnn_branch)
    
    # Bidirectional LSTM branch for temporal patterns
    lstm_branch = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(input_layer)
    lstm_branch = BatchNormalization()(lstm_branch)
    lstm_branch = Bidirectional(LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(lstm_branch)
    
    # GRU branch for different temporal modeling
    gru_branch = Bidirectional(GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(input_layer)
    gru_branch = BatchNormalization()(gru_branch)
    gru_branch = Bidirectional(GRU(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(gru_branch)
    
    # Combine branches
    combined = Concatenate()([cnn_branch, lstm_branch, gru_branch])
    
    # Global pooling and dense layers
    global_pool = GlobalAveragePooling1D()(combined)
    
    dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(global_pool)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.5)(dense2)
    
    dense3 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense2)
    dense3 = Dropout(0.3)(dense3)
    
    output = Dense(2, activation='softmax')(dense3)
    
    model = Model(inputs=input_layer, outputs=output)
    return model

def main():
    print("Loading enhanced dataset...")
    X_seq, y = load_enhanced_data(use_augmentation=True)
    
    print(f"Loaded {len(X_seq)} samples")
    print(f"Label distribution: {Counter(y)}")
    
    # Pad sequences
    X_padded = pad_sequences(X_seq, maxlen=MAX_LEN, dtype='float32', padding='post', truncating='post')
    print(f"Feature shape: {X_padded.shape}")
    
    # Normalize features
    n_samples, n_timesteps, n_features = X_padded.shape
    X_reshaped = X_padded.reshape(-1, n_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Save scaler
    joblib.dump(scaler, "feature_scaler.pkl")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Save label encoder
    joblib.dump(le, "label_encoder.pkl")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_encoded), 
        y=y_encoded
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weights_dict}")
    
    # Create model
    model = create_advanced_model((MAX_LEN, X_scaled.shape[2]))
    
    # Compile model
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'precision', 'recall']
    )
    
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        "parkinsons_voice_model_best.h5", 
        monitor='val_accuracy', 
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )
    
    # Save final model
    model.save("parkinsons_voice_model_final.h5")
    print("Model saved successfully.")
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc, test_prec, test_recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Predictions and detailed metrics
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Training history plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Cross-validation for more robust evaluation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_encoded)):
        print(f"Fold {fold + 1}/5")
        
        X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
        y_train_cv, y_val_cv = y_categorical[train_idx], y_categorical[val_idx]
        
        # Create new model for each fold
        model_cv = create_advanced_model((MAX_LEN, X_scaled.shape[2]))
        model_cv.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        # Train with early stopping
        early_stop_cv = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model_cv.fit(
            X_train_cv, y_train_cv,
            epochs=50,
            batch_size=16,
            validation_data=(X_val_cv, y_val_cv),
            callbacks=[early_stop_cv],
            verbose=0
        )
        
        # Evaluate
        _, cv_acc = model_cv.evaluate(X_val_cv, y_val_cv, verbose=0)
        cv_scores.append(cv_acc)
        print(f"Fold {fold + 1} Accuracy: {cv_acc:.4f}")
    
    print(f"\nCross-validation Results:")
    print(f"Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return model, history, le, scaler

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model, history, le, scaler = main()
