import os
import numpy as np
import librosa
import joblib
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Constants
SAMPLE_RATE = 16000  # Wav2Vec2 expects 16kHz
DURATION = 3.0  # seconds

class Wav2VecFeatureExtractor:
    def __init__(self, model_name="facebook/wav2vec2-base"):
        """Initialize Wav2Vec2 feature extractor and model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def extract_features(self, audio_path):
        """Extract Wav2Vec2 features from audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION, offset=0.5)
            
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
                temporal_features = np.mean(features, axis=0)
                combined_features = np.concatenate([temporal_features, feature_stats])
                
                return combined_features
                
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

def predict_parkinson_wav2vec(audio_file_path, model_path="wav2vec_best_model.pkl"):
    """
    Predict Parkinson's disease from an audio file using Wav2Vec2 features
    
    Args:
        audio_file_path (str): Path to the audio file
        model_path (str): Path to the trained model
    
    Returns:
        dict: Prediction results including probability and classification
    """
    try:
        # Load model and preprocessors
        model = joblib.load(model_path)
        le = joblib.load("wav2vec_label_encoder.pkl")
        scaler = joblib.load("wav2vec_scaler.pkl")
        
        # Initialize feature extractor
        feature_extractor = Wav2VecFeatureExtractor()
        
        # Extract features
        features = feature_extractor.extract_features(audio_file_path)
        if features is None:
            return {"error": "Failed to extract features from audio file"}
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probabilities if available
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(features_scaled)[0]
            confidence = np.max(prediction_proba)
        else:
            prediction_proba = None
            confidence = 1.0  # For models without probability estimates
        
        # Get class names
        class_names = le.classes_
        predicted_class = class_names[prediction]
        
        # Probability for each class
        if prediction_proba is not None:
            class_probabilities = {
                class_names[i]: float(prediction_proba[i]) 
                for i in range(len(class_names))
            }
            parkinson_prob = float(class_probabilities.get("PD_AH", 0.0))
            healthy_prob = float(class_probabilities.get("HC_AH", 0.0))
        else:
            # For models without probability estimates
            class_probabilities = {predicted_class: 1.0}
            for cls in class_names:
                if cls != predicted_class:
                    class_probabilities[cls] = 0.0
            parkinson_prob = float(class_probabilities.get("PD_AH", 0.0))
            healthy_prob = float(class_probabilities.get("HC_AH", 0.0))
        
        result = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "class_probabilities": class_probabilities,
            "parkinson_probability": parkinson_prob,
            "healthy_probability": healthy_prob,
            "risk_level": get_risk_level(parkinson_prob),
            "model_type": "Wav2Vec2-based"
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def predict_with_all_models(audio_file_path):
    """
    Predict using all available trained models and return ensemble result
    """
    model_files = [
        ("Random Forest", "wav2vec_model_random_forest.pkl"),
        ("Gradient Boosting", "wav2vec_model_gradient_boosting.pkl"),
        ("SVM", "wav2vec_model_svm.pkl"),
        ("Neural Network", "wav2vec_model_neural_network.pkl"),
        ("Logistic Regression", "wav2vec_model_logistic_regression.pkl")
    ]
    
    predictions = {}
    parkinson_probs = []
    
    for model_name, model_file in model_files:
        if os.path.exists(model_file):
            result = predict_parkinson_wav2vec(audio_file_path, model_file)
            if "error" not in result:
                predictions[model_name] = result
                parkinson_probs.append(result["parkinson_probability"])
    
    if not predictions:
        return {"error": "No trained models found"}
    
    # Ensemble prediction
    ensemble_parkinson_prob = np.mean(parkinson_probs)
    ensemble_prediction = "PD_AH" if ensemble_parkinson_prob > 0.5 else "HC_AH"
    
    ensemble_result = {
        "ensemble_predicted_class": ensemble_prediction,
        "ensemble_parkinson_probability": float(ensemble_parkinson_prob),
        "ensemble_healthy_probability": float(1 - ensemble_parkinson_prob),
        "ensemble_risk_level": get_risk_level(ensemble_parkinson_prob),
        "individual_predictions": predictions,
        "model_agreement": len([p for p in parkinson_probs if (p > 0.5) == (ensemble_parkinson_prob > 0.5)]) / len(parkinson_probs)
    }
    
    return ensemble_result

def get_risk_level(parkinson_prob):
    """Determine risk level based on Parkinson's probability"""
    if parkinson_prob < 0.3:
        return "Low Risk"
    elif parkinson_prob < 0.6:
        return "Moderate Risk"
    elif parkinson_prob < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"

def batch_predict(audio_folder_path, model_path="wav2vec_best_model.pkl"):
    """
    Predict Parkinson's disease for multiple audio files in a folder
    
    Args:
        audio_folder_path (str): Path to folder containing audio files
        model_path (str): Path to the trained model
    
    Returns:
        list: List of prediction results for each file
    """
    results = []
    
    for filename in os.listdir(audio_folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(audio_folder_path, filename)
            result = predict_parkinson_wav2vec(file_path, model_path)
            result["filename"] = filename
            results.append(result)
    
    return results

def print_prediction_report(result):
    """Print a formatted prediction report"""
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print("=" * 50)
    print("PARKINSON'S DISEASE PREDICTION REPORT")
    print("=" * 50)
    
    if "ensemble_predicted_class" in result:
        # Ensemble result
        print("=== ENSEMBLE PREDICTION ===")
        print(f"Predicted Class: {result['ensemble_predicted_class']}")
        print(f"Parkinson's Probability: {result['ensemble_parkinson_probability']:.3f}")
        print(f"Healthy Probability: {result['ensemble_healthy_probability']:.3f}")
        print(f"Risk Level: {result['ensemble_risk_level']}")
        print(f"Model Agreement: {result['model_agreement']:.3f}")
        print("\n=== INDIVIDUAL MODEL PREDICTIONS ===")
        
        for model_name, pred in result['individual_predictions'].items():
            print(f"{model_name}:")
            print(f"  Prediction: {pred['predicted_class']}")
            print(f"  Parkinson's Prob: {pred['parkinson_probability']:.3f}")
            print(f"  Confidence: {pred['confidence']:.3f}")
    else:
        # Single model result
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Model Type: {result.get('model_type', 'Unknown')}")
        print("-" * 30)
        print("Class Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            status = "Healthy Control" if class_name == "HC_AH" else "Parkinson's Disease"
            print(f"  {status}: {prob:.3f}")
    
    print("=" * 50)

if __name__ == "__main__":
    # Example usage
    test_file = "../data/TestData/shashwat.wav"  # Update this path
    
    if os.path.exists(test_file):
        print(f"Analyzing: {test_file}")
        
        # Single model prediction
        print("\n--- Single Model Prediction ---")
        result = predict_parkinson_wav2vec(test_file)
        print_prediction_report(result)
        
        # Ensemble prediction (if multiple models are available)
        print("\n--- Ensemble Prediction ---")
        ensemble_result = predict_with_all_models(test_file)
        print_prediction_report(ensemble_result)
        
    else:
        print(f"Test file not found: {test_file}")
        print("Please update the test_file path to point to a valid audio file.")
