import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import warnings
warnings.filterwarnings('ignore')

class ParkinsonsPredictor:
    """
    A comprehensive Parkinson's disease prediction system using Wav2Vec2 features
    """
    
    def __init__(self, models_dir=".", wav2vec_model="facebook/wav2vec2-base"):
        """
        Initialize the predictor with trained models
        
        Args:
            models_dir (str): Directory containing the trained model files
            wav2vec_model (str): Wav2Vec2 model name from HuggingFace
        """
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize Wav2Vec2
        print("Loading Wav2Vec2 model...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_model)
        self.wav2vec_model.to(self.device)
        self.wav2vec_model.eval()
        
        # Load trained models and preprocessors
        self.load_models()
        
    def load_models(self):
        """Load all trained models and preprocessors"""
        try:
            # Load preprocessors
            self.scaler = joblib.load(os.path.join(self.models_dir, "wav2vec_scaler.pkl"))
            self.label_encoder = joblib.load(os.path.join(self.models_dir, "wav2vec_label_encoder.pkl"))
            print("✓ Loaded preprocessors")
            
            # Load individual models
            self.models = {}
            model_files = [
                ("Random Forest", "wav2vec_model_random_forest.pkl"),
                ("Gradient Boosting", "wav2vec_model_gradient_boosting.pkl"),
                ("SVM", "wav2vec_model_svm.pkl"),
                ("Neural Network", "wav2vec_model_neural_network.pkl"),
                ("Logistic Regression", "wav2vec_model_logistic_regression.pkl"),
                ("Best Model", "wav2vec_best_model.pkl")
            ]
            
            for model_name, filename in model_files:
                model_path = os.path.join(self.models_dir, filename)
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    print(f"✓ Loaded {model_name}")
                else:
                    print(f"⚠ Model not found: {filename}")
            
            if not self.models:
                raise FileNotFoundError("No trained models found!")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def extract_wav2vec_features(self, audio_path):
        """
        Extract Wav2Vec2 features from audio file
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            numpy.ndarray: Extracted features
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, duration=3.0, offset=0.5)
            
            # Ensure consistent length
            target_length = int(16000 * 3.0)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            
            # Extract features using Wav2Vec2
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.wav2vec_model(**inputs)
                
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
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def predict_single_file(self, audio_path, use_ensemble=True):
        """
        Predict Parkinson's disease for a single audio file
        
        Args:
            audio_path (str): Path to audio file
            use_ensemble (bool): Whether to use ensemble prediction
            
        Returns:
            dict: Prediction results
        """
        if not os.path.exists(audio_path):
            return {"error": f"File not found: {audio_path}"}
        
        print(f"Analyzing: {audio_path}")
        
        # Extract features
        features = self.extract_wav2vec_features(audio_path)
        if features is None:
            return {"error": "Failed to extract features"}
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Get predictions from all models
        predictions = {}
        parkinson_probs = []
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(features_scaled)[0]
                
                if hasattr(model, "predict_proba"):
                    pred_proba = model.predict_proba(features_scaled)[0]
                    confidence = np.max(pred_proba)
                    class_probs = {
                        self.label_encoder.classes_[i]: float(pred_proba[i])
                        for i in range(len(self.label_encoder.classes_))
                    }
                else:
                    confidence = 1.0
                    class_probs = {cls: 1.0 if cls == self.label_encoder.classes_[pred] else 0.0 
                                 for cls in self.label_encoder.classes_}
                
                parkinson_prob = class_probs.get("PD_AH", 0.0)
                parkinson_probs.append(parkinson_prob)
                
                predictions[model_name] = {
                    "predicted_class": self.label_encoder.classes_[pred],
                    "confidence": confidence,
                    "parkinson_probability": parkinson_prob,
                    "healthy_probability": class_probs.get("HC_AH", 0.0),
                    "class_probabilities": class_probs
                }
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue
        
        if not predictions:
            return {"error": "No models could make predictions"}
        
        # Calculate ensemble results
        if use_ensemble and len(parkinson_probs) > 1:
            ensemble_parkinson_prob = np.mean(parkinson_probs)
            ensemble_std = np.std(parkinson_probs)
            ensemble_prediction = "PD_AH" if ensemble_parkinson_prob > 0.5 else "HC_AH"
            
            # Model agreement
            binary_preds = [1 if p > 0.5 else 0 for p in parkinson_probs]
            agreement = len([p for p in binary_preds if p == (1 if ensemble_parkinson_prob > 0.5 else 0)]) / len(binary_preds)
            
            result = {
                "filename": os.path.basename(audio_path),
                "ensemble_prediction": ensemble_prediction,
                "ensemble_parkinson_probability": float(ensemble_parkinson_prob),
                "ensemble_healthy_probability": float(1 - ensemble_parkinson_prob),
                "prediction_uncertainty": float(ensemble_std),
                "model_agreement": float(agreement),
                "risk_level": self.get_risk_level(ensemble_parkinson_prob),
                "individual_predictions": predictions,
                "recommendation": self.get_recommendation(ensemble_parkinson_prob, agreement)
            }
        else:
            # Use best single model
            best_model = "Best Model" if "Best Model" in predictions else list(predictions.keys())[0]
            best_pred = predictions[best_model]
            
            result = {
                "filename": os.path.basename(audio_path),
                "prediction": best_pred["predicted_class"],
                "parkinson_probability": best_pred["parkinson_probability"],
                "healthy_probability": best_pred["healthy_probability"],
                "confidence": best_pred["confidence"],
                "risk_level": self.get_risk_level(best_pred["parkinson_probability"]),
                "model_used": best_model,
                "recommendation": self.get_recommendation(best_pred["parkinson_probability"], 1.0)
            }
        
        return result
    
    def predict_multiple_files(self, folder_path, output_csv=None):
        """
        Predict Parkinson's disease for multiple audio files
        
        Args:
            folder_path (str): Path to folder containing audio files
            output_csv (str): Optional path to save results as CSV
            
        Returns:
            list: List of prediction results
        """
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return []
        
        audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
        
        if not audio_files:
            print("No audio files found in the folder")
            return []
        
        print(f"Found {len(audio_files)} audio files")
        results = []
        
        for i, filename in enumerate(audio_files, 1):
            file_path = os.path.join(folder_path, filename)
            print(f"\n[{i}/{len(audio_files)}] Processing: {filename}")
            
            result = self.predict_single_file(file_path)
            if "error" not in result:
                results.append(result)
                
                # Print summary
                if "ensemble_prediction" in result:
                    print(f"  → Ensemble: {result['ensemble_prediction']} ({result['ensemble_parkinson_probability']:.3f})")
                    print(f"  → Risk: {result['risk_level']} | Agreement: {result['model_agreement']:.2f}")
                else:
                    print(f"  → Prediction: {result['prediction']} ({result['parkinson_probability']:.3f})")
                    print(f"  → Risk: {result['risk_level']}")
            else:
                print(f"  → Error: {result['error']}")
        
        # Save to CSV if requested
        if output_csv and results:
            self.save_results_to_csv(results, output_csv)
        
        return results
    
    def save_results_to_csv(self, results, output_path):
        """Save prediction results to CSV file"""
        try:
            rows = []
            for result in results:
                if "ensemble_prediction" in result:
                    row = {
                        "filename": result["filename"],
                        "prediction": result["ensemble_prediction"],
                        "parkinson_probability": result["ensemble_parkinson_probability"],
                        "healthy_probability": result["ensemble_healthy_probability"],
                        "risk_level": result["risk_level"],
                        "model_agreement": result["model_agreement"],
                        "uncertainty": result["prediction_uncertainty"],
                        "recommendation": result["recommendation"]
                    }
                else:
                    row = {
                        "filename": result["filename"],
                        "prediction": result["prediction"],
                        "parkinson_probability": result["parkinson_probability"],
                        "healthy_probability": result["healthy_probability"],
                        "risk_level": result["risk_level"],
                        "model_agreement": 1.0,
                        "uncertainty": 0.0,
                        "recommendation": result["recommendation"]
                    }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            print(f"\n✓ Results saved to: {output_path}")
            
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    
    def get_risk_level(self, parkinson_prob):
        """Determine risk level based on Parkinson's probability"""
        if parkinson_prob < 0.3:
            return "Low Risk"
        elif parkinson_prob < 0.6:
            return "Moderate Risk"
        elif parkinson_prob < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def get_recommendation(self, parkinson_prob, agreement):
        """Get clinical recommendation based on prediction"""
        if parkinson_prob < 0.3 and agreement > 0.7:
            return "Low concern. Continue regular monitoring."
        elif parkinson_prob < 0.6:
            return "Moderate concern. Consider consultation with neurologist."
        elif parkinson_prob < 0.8:
            return "High concern. Recommend neurological evaluation."
        else:
            return "Very high concern. Urgent neurological consultation recommended."
    
    def print_detailed_report(self, result):
        """Print a detailed prediction report"""
        print("\n" + "="*60)
        print("PARKINSON'S DISEASE PREDICTION REPORT")
        print("="*60)
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
            return
        
        print(f"File: {result['filename']}")
        
        if "ensemble_prediction" in result:
            print(f"\n🎯 ENSEMBLE PREDICTION")
            print(f"   Predicted Class: {result['ensemble_prediction']}")
            print(f"   Parkinson's Probability: {result['ensemble_parkinson_probability']:.3f}")
            print(f"   Healthy Probability: {result['ensemble_healthy_probability']:.3f}")
            print(f"   Model Agreement: {result['model_agreement']:.3f}")
            print(f"   Prediction Uncertainty: {result['prediction_uncertainty']:.3f}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"\n💡 RECOMMENDATION")
            print(f"   {result['recommendation']}")
            
            print(f"\n📊 INDIVIDUAL MODEL PREDICTIONS")
            for model_name, pred in result['individual_predictions'].items():
                print(f"   {model_name}:")
                print(f"     • Prediction: {pred['predicted_class']}")
                print(f"     • Parkinson's Prob: {pred['parkinson_probability']:.3f}")
                print(f"     • Confidence: {pred['confidence']:.3f}")
        else:
            print(f"\n🎯 PREDICTION ({result['model_used']})")
            print(f"   Predicted Class: {result['prediction']}")
            print(f"   Parkinson's Probability: {result['parkinson_probability']:.3f}")
            print(f"   Healthy Probability: {result['healthy_probability']:.3f}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"\n💡 RECOMMENDATION")
            print(f"   {result['recommendation']}")
        
        print("="*60)

def main():
    """Main function for command-line usage"""
    print("🎤 Parkinson's Disease Voice Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    try:
        predictor = ParkinsonsPredictor()
        print("✓ Models loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return
    
    while True:
        print("\nOptions:")
        print("1. Predict single audio file")
        print("2. Predict multiple files from folder")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            audio_path = input("Enter path to audio file: ").strip().strip('"')
            if audio_path:
                result = predictor.predict_single_file(audio_path)
                predictor.print_detailed_report(result)
        
        elif choice == "2":
            folder_path = input("Enter path to folder containing audio files: ").strip().strip('"')
            save_csv = input("Save results to CSV? (y/n): ").strip().lower() == 'y'
            
            if folder_path:
                output_csv = None
                if save_csv:
                    output_csv = input("Enter CSV filename (or press Enter for default): ").strip()
                    if not output_csv:
                        output_csv = "prediction_results.csv"
                
                results = predictor.predict_multiple_files(folder_path, output_csv)
                
                if results:
                    print(f"\n📈 SUMMARY OF {len(results)} PREDICTIONS")
                    print("-" * 40)
                    
                    risk_counts = {}
                    for result in results:
                        risk = result.get('risk_level', 'Unknown')
                        risk_counts[risk] = risk_counts.get(risk, 0) + 1
                    
                    for risk, count in risk_counts.items():
                        print(f"{risk}: {count} files")
        
        elif choice == "3":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
