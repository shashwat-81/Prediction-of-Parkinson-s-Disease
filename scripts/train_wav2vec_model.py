import os
import numpy as np
import librosa
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Constants
DATA_PATH = r'C:\Users\mishr\OneDrive\Desktop\Prediction-of-Parkinson-s-Disease\data\voice'
SAMPLE_RATE = 16000  # Wav2Vec2 expects 16kHz
DURATION = 3.0  # seconds

class Wav2VecFeatureExtractor:
    def __init__(self, model_name="facebook/wav2vec2-base"):
        """Initialize Wav2Vec2 feature extractor and model"""
        print(f"Loading Wav2Vec2 model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def extract_features(self, audio_path, augment=False):
        """Extract Wav2Vec2 features from audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION, offset=0.5)
            
            # Data augmentation if requested
            if augment:
                # Random noise
                if np.random.random() > 0.5:
                    noise = np.random.normal(0, 0.01, audio.shape)
                    audio = audio + noise
                
                # Time stretching
                if np.random.random() > 0.5:
                    stretch_rate = np.random.uniform(0.9, 1.1)
                    audio = librosa.effects.time_stretch(audio, rate=stretch_rate)
                    # Ensure consistent length
                    if len(audio) > SAMPLE_RATE * DURATION:
                        audio = audio[:int(SAMPLE_RATE * DURATION)]
                    elif len(audio) < SAMPLE_RATE * DURATION:
                        audio = np.pad(audio, (0, int(SAMPLE_RATE * DURATION) - len(audio)))
                
                # Pitch shifting
                if np.random.random() > 0.5:
                    n_steps = np.random.uniform(-1, 1)
                    audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)
            
            # Ensure consistent length
            target_length = int(SAMPLE_RATE * DURATION)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            
            # Extract features using Wav2Vec2
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                
                # Get the last hidden state
                features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                
                # Additional statistical features
                feature_stats = np.array([
                    np.mean(features, axis=0),
                    np.std(features, axis=0),
                    np.max(features, axis=0),
                    np.min(features, axis=0),
                    np.percentile(features, 25, axis=0),
                    np.percentile(features, 75, axis=0)
                ]).flatten()
                
                # Combine temporal features with statistical features
                # Use mean pooling over time dimension for temporal features
                temporal_features = np.mean(features, axis=0)
                combined_features = np.concatenate([temporal_features, feature_stats])
                
                return combined_features
                
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

def load_wav2vec_data(feature_extractor, use_augmentation=True):
    """Load data with Wav2Vec2 features"""
    features = []
    labels = []
    filenames = []

    for label in ['HC_AH', 'PD_AH']:
        folder = os.path.join(DATA_PATH, label)
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        
        print(f"Processing {len(files)} files from {label}...")
        
        for i, file_name in enumerate(files):
            file_path = os.path.join(folder, file_name)
            
            # Original features
            feature_vec = feature_extractor.extract_features(file_path, augment=False)
            if feature_vec is not None:
                features.append(feature_vec)
                labels.append(label)
                filenames.append(file_name)
            
            # Augmented features (create 2 augmented versions per original)
            if use_augmentation:
                for aug_idx in range(2):
                    aug_feature_vec = feature_extractor.extract_features(file_path, augment=True)
                    if aug_feature_vec is not None:
                        features.append(aug_feature_vec)
                        labels.append(label)
                        filenames.append(f"{file_name}_aug_{aug_idx}")
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(files)} files")
    
    return np.array(features), np.array(labels), filenames

def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            roc_auc = None
        
        results[name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        trained_models[name] = model
        
        roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc_str}")
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))
        print("-" * 50)
    
    return results, trained_models

def perform_cross_validation(X, y, best_model, cv_folds=5):
    """Perform cross-validation for robust evaluation"""
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    cv_roc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        # Clone and train model
        model_cv = type(best_model)(**best_model.get_params())
        model_cv.fit(X_train_cv, y_train_cv)
        
        # Evaluate
        y_pred_cv = model_cv.predict(X_val_cv)
        y_pred_proba_cv = model_cv.predict_proba(X_val_cv) if hasattr(model_cv, "predict_proba") else None
        
        cv_acc = accuracy_score(y_val_cv, y_pred_cv)
        cv_scores.append(cv_acc)
        
        if y_pred_proba_cv is not None:
            cv_roc = roc_auc_score(y_val_cv, y_pred_proba_cv[:, 1])
            cv_roc_scores.append(cv_roc)
        
        print(f"Fold {fold + 1}: Accuracy = {cv_acc:.4f}")
    
    print(f"\nCross-validation Results:")
    print(f"Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    if cv_roc_scores:
        print(f"Mean ROC AUC: {np.mean(cv_roc_scores):.4f} ± {np.std(cv_roc_scores):.4f}")
    
    return cv_scores, cv_roc_scores

def plot_results(results, y_test, le):
    """Plot comparison results"""
    # Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    roc_aucs = [results[name]['roc_auc'] if results[name]['roc_auc'] else 0 for name in model_names]
    
    axes[0, 0].bar(model_names, accuracies)
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(model_names, roc_aucs)
    axes[0, 1].set_title('Model ROC AUC Comparison')
    axes[0, 1].set_ylabel('ROC AUC')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Confusion matrices for best models
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    y_pred_best = results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1, 0])
    axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Feature importance (if available)
    if hasattr(results[best_model_name], 'feature_importances_'):
        feature_importance = results[best_model_name].feature_importances_
        indices = np.argsort(feature_importance)[-20:]  # Top 20 features
        axes[1, 1].barh(range(len(indices)), feature_importance[indices])
        axes[1, 1].set_title('Top 20 Feature Importance')
        axes[1, 1].set_xlabel('Importance')
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Feature Importance')
    
    plt.tight_layout()
    plt.show()
    
    return best_model_name

def main():
    """Main training function"""
    print("Starting Wav2Vec2-based Parkinson's Disease Detection Training...")
    
    # Initialize Wav2Vec2 feature extractor
    feature_extractor = Wav2VecFeatureExtractor()
    
    # Load data with Wav2Vec2 features
    print("\nLoading and processing data...")
    X, y, filenames = load_wav2vec_data(feature_extractor, use_augmentation=True)
    
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Label distribution: {Counter(y)}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    # Train multiple models
    results, trained_models = train_multiple_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = trained_models[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {results[best_model_name]['accuracy']:.4f}")
    best_roc_str = f"{results[best_model_name]['roc_auc']:.4f}" if results[best_model_name]['roc_auc'] is not None else "N/A"
    print(f"Best ROC AUC: {best_roc_str}")
    
    # Cross-validation
    perform_cross_validation(X, y_encoded, best_model)
    
    # Plot results
    plot_results(results, y_test, le)
    
    # Save models and preprocessors
    print("\nSaving models and preprocessors...")
    joblib.dump(best_model, "wav2vec_best_model.pkl")
    joblib.dump(scaler, "wav2vec_scaler.pkl")
    joblib.dump(le, "wav2vec_label_encoder.pkl")
    
    # Save all models
    for name, model in trained_models.items():
        filename = f"wav2vec_model_{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
    
    print("Training completed successfully!")
    
    return best_model, scaler, le, feature_extractor

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run training
    best_model, scaler, le, feature_extractor = main()
