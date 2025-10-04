import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class MedicalImagePreprocessor:
    """Medical-grade image preprocessing for prediction"""
    
    @staticmethod
    def extract_drawing_features(image_path):
        """Extract clinically relevant features from drawing images"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Denoise
        img_denoised = cv2.fastNlMeansDenoising(img)
        
        # Edge detection for line analysis
        edges = cv2.Canny(img_denoised, 50, 150)
        
        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {}
        
        if contours:
            # Get main contour (largest)
            main_contour = max(contours, key=cv2.contourArea)
            
            # 1. Tremor Analysis - Line smoothness
            epsilon = 0.02 * cv2.arcLength(main_contour, True)
            approx = cv2.approxPolyDP(main_contour, epsilon, True)
            features['line_smoothness'] = len(approx) / len(main_contour) if len(main_contour) > 0 else 0
            
            # 2. Micrographia - Drawing size consistency
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(main_contour)
            features['size_consistency'] = contour_area / hull_area if hull_area > 0 else 0
            
            # 3. Bradykinesia - Line thickness variation
            features['thickness_variation'] = np.std(edges.flatten()) / 255.0
            
            # 4. Rigidity - Angular changes
            if len(main_contour) > 10:
                angles = []
                for i in range(len(main_contour)):
                    p1 = main_contour[i-1][0] if i > 0 else main_contour[-1][0]
                    p2 = main_contour[i][0]
                    p3 = main_contour[i+1][0] if i < len(main_contour)-1 else main_contour[0][0]
                    
                    v1 = p1 - p2
                    v2 = p3 - p2
                    
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cosine = np.clip(cosine, -1, 1)
                        angle = np.arccos(cosine)
                        angles.append(angle)
                
                features['angular_consistency'] = 1 - (np.std(angles) / np.pi) if angles else 0
            else:
                features['angular_consistency'] = 0
        else:
            features = {
                'line_smoothness': 0,
                'size_consistency': 0,
                'thickness_variation': 0,
                'angular_consistency': 0
            }
        
        return features
    
    @staticmethod
    def enhance_medical_features(image):
        """Enhance medically relevant features in the image"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        # Enhance contrast for better line detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_gray)
        
        # Gaussian blur to reduce noise while preserving edges
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Convert back to PIL and RGB
        enhanced_pil = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB))
        
        return enhanced_pil

class MedicalVisionTransformer(nn.Module):
    """Medical-grade Vision Transformer with clinical feature fusion"""
    
    def __init__(self, num_classes=2, num_medical_features=4):
        super(MedicalVisionTransformer, self).__init__()
        
        # Load pre-trained ViT
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Remove the original classifier
        vit_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()
        
        # Medical feature processor
        self.medical_processor = nn.Sequential(
            nn.Linear(num_medical_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(vit_features + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(embed_dim=vit_features + 128, num_heads=8, dropout=0.1)
    
    def forward(self, images, medical_features):
        # Extract visual features
        visual_features = self.vit(images)
        
        # Process medical features
        medical_processed = self.medical_processor(medical_features)
        
        # Concatenate features
        combined_features = torch.cat([visual_features, medical_processed], dim=1)
        
        # Apply attention
        combined_features = combined_features.unsqueeze(0)  # Add sequence dimension
        attended_features, _ = self.attention(combined_features, combined_features, combined_features)
        attended_features = attended_features.squeeze(0)  # Remove sequence dimension
        
        # Final classification
        output = self.fusion(attended_features)
        
        return output

class MedicalParkinsonPredictor:
    """Complete prediction system for Parkinson's disease detection"""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the medical predictor
        
        Args:
            model_path (str): Path to trained model file (.pth)
            device (str): 'cuda' or 'cpu'
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_info = None
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"🏥 Medical Parkinson's Predictor Initialized")
        print(f"📱 Device: {self.device}")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("⚠️ No model loaded. Use load_model() to load a trained model.")
    
    def load_model(self, model_path):
        """Load trained medical model"""
        try:
            print(f"📂 Loading model from: {model_path}")

            # Attempt to load checkpoint. Newer PyTorch versions (>=2.6)
            # default to weights-only loading which blocks some globals.
            # We'll attempt a safe progressive strategy:
            # 1) Try a normal torch.load (uses default behavior).
            # 2) If it fails with a WeightsUnpickler/global error, try to
            #    allowlist the known numpy scalar global if available.
            # 3) As a last resort (if user trusts the file), re-load with
            #    weights_only=False which allows arbitrary pickles.
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception as load_err:
                err_msg = str(load_err)
                print(f"⚠️ Initial torch.load failed: {err_msg}")

                # Try to allowlist numpy._core.multiarray.scalar if available
                tried_allowlist = False
                try:
                    if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                        import numpy as _np
                        # Some numpy builds expose the scalar in a few locations
                        scalar_obj = None
                        try:
                            scalar_obj = _np._core.multiarray.scalar
                        except Exception:
                            # Fallback: try to get from numpy.core.multiarray
                            try:
                                scalar_obj = _np.core.multiarray.scalar
                            except Exception:
                                scalar_obj = None

                        if scalar_obj is not None:
                            print("🔐 Adding numpy scalar to torch safe globals and retrying load...")
                            torch.serialization.add_safe_globals([scalar_obj])
                            tried_allowlist = True
                            checkpoint = torch.load(model_path, map_location=self.device)
                        else:
                            print("⚠️ Could not locate numpy scalar object to allowlist")
                except Exception as allow_err:
                    print(f"⚠️ Allowlist attempt failed: {allow_err}")

                # If we didn't get a checkpoint yet, try the less-safe weights_only=False
                if 'checkpoint' not in locals():
                    try:
                        print("⚠️ Falling back to torch.load(..., weights_only=False)."
                              " This may execute arbitrary code from the checkpoint."
                              " Only do this if you trust the model file source.")
                        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    except Exception as final_err:
                        print(f"❌ Final load attempt failed: {final_err}")
                        raise final_err
            self.model_info = {
                'dataset_type': checkpoint.get('dataset_type', 'spiral'),
                'medical_features': checkpoint.get('medical_features', 
                    ['line_smoothness', 'size_consistency', 'thickness_variation', 'angular_consistency']),
                'training_history': checkpoint.get('training_history', {})
            }
            
            # Initialize model
            self.model = MedicalVisionTransformer(num_classes=2, num_medical_features=4)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"   • Dataset type: {self.model_info['dataset_type']}")
            print(f"   • Medical features: {len(self.model_info['medical_features'])}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_single_image(self, image_path, return_details=True):
        """
        Predict Parkinson's disease for a single image
        
        Args:
            image_path (str): Path to the drawing image
            return_details (bool): Return detailed analysis
            
        Returns:
            dict: Prediction results with clinical interpretation
        """
        if self.model is None:
            raise ValueError("No model loaded! Use load_model() first.")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"\n🔬 Analyzing: {os.path.basename(image_path)}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            enhanced_image = MedicalImagePreprocessor.enhance_medical_features(image)
            
            # Extract medical features
            medical_features_dict = MedicalImagePreprocessor.extract_drawing_features(image_path)
            if medical_features_dict is None:
                raise ValueError("Could not extract features from image")
            
            # Prepare tensors
            image_tensor = self.transform(enhanced_image).unsqueeze(0).to(self.device)
            medical_tensor = torch.tensor([
                medical_features_dict['line_smoothness'],
                medical_features_dict['size_consistency'],
                medical_features_dict['thickness_variation'],
                medical_features_dict['angular_consistency']
            ], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor, medical_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1)
            
            # Extract results
            healthy_prob = probabilities[0][0].item()
            parkinson_prob = probabilities[0][1].item()
            prediction = predicted_class.item()
            
            # Clinical interpretation
            confidence = max(healthy_prob, parkinson_prob)
            result = {
                'prediction': 'Parkinson\'s Disease' if prediction == 1 else 'Healthy',
                'prediction_class': prediction,
                'parkinson_probability': parkinson_prob,
                'healthy_probability': healthy_prob,
                'confidence': confidence,
                'medical_features': medical_features_dict,
                'clinical_interpretation': self._get_clinical_interpretation(
                    parkinson_prob, medical_features_dict, confidence
                ),
                'image_path': image_path
            }
            
            # Print results
            self._print_prediction_results(result)
            
            if return_details:
                return result
            else:
                return {
                    'prediction': result['prediction'],
                    'parkinson_probability': result['parkinson_probability'],
                    'confidence': result['confidence']
                }
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def predict_batch(self, image_paths, save_results=True):
        """
        Predict multiple images at once
        
        Args:
            image_paths (list): List of image paths
            save_results (bool): Save results to CSV
            
        Returns:
            list: List of prediction results
        """
        print(f"\n🔬 Batch Analysis: {len(image_paths)} images")
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n📊 Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            result = self.predict_single_image(image_path, return_details=True)
            if result:
                results.append(result)
        
        # Summary statistics
        if results:
            parkinson_count = sum(1 for r in results if r['prediction_class'] == 1)
            healthy_count = len(results) - parkinson_count
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            print(f"\n📊 Batch Summary:")
            print(f"   • Total analyzed: {len(results)}")
            print(f"   • Parkinson's detected: {parkinson_count} ({parkinson_count/len(results)*100:.1f}%)")
            print(f"   • Healthy: {healthy_count} ({healthy_count/len(results)*100:.1f}%)")
            print(f"   • Average confidence: {avg_confidence:.3f}")
        
        # Save results
        if save_results and results:
            self._save_batch_results(results)
        
        return results
    
    def analyze_drawing_folder(self, folder_path, file_extensions=('.png', '.jpg', '.jpeg')):
        """
        Analyze all drawings in a folder
        
        Args:
            folder_path (str): Path to folder containing drawings
            file_extensions (tuple): Allowed file extensions
            
        Returns:
            list: Analysis results
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(Path(folder_path).glob(f"*{ext}"))
            image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No images found in {folder_path}")
            return []
        
        image_paths = [str(img) for img in image_files]
        print(f"📁 Found {len(image_paths)} images in folder")
        
        return self.predict_batch(image_paths)
    
    def _get_clinical_interpretation(self, parkinson_prob, medical_features, confidence):
        """Generate clinical interpretation"""
        interpretation = {
            'risk_level': '',
            'clinical_notes': [],
            'recommendations': []
        }
        
        # Risk assessment
        if parkinson_prob >= 0.8:
            interpretation['risk_level'] = 'HIGH'
            interpretation['clinical_notes'].append("Strong indicators of Parkinson's motor symptoms")
            interpretation['recommendations'].append("Immediate neurological consultation recommended")
        elif parkinson_prob >= 0.6:
            interpretation['risk_level'] = 'MODERATE'
            interpretation['clinical_notes'].append("Some indicators of motor impairment present")
            interpretation['recommendations'].append("Follow-up assessment recommended")
        elif parkinson_prob >= 0.4:
            interpretation['risk_level'] = 'LOW-MODERATE'
            interpretation['clinical_notes'].append("Minimal indicators present")
            interpretation['recommendations'].append("Monitor symptoms, routine follow-up")
        else:
            interpretation['risk_level'] = 'LOW'
            interpretation['clinical_notes'].append("No significant motor impairment detected")
            interpretation['recommendations'].append("Continue regular health monitoring")
        
        # Feature-specific notes
        if medical_features['line_smoothness'] < 0.3:
            interpretation['clinical_notes'].append("Tremor indicators detected in line execution")
        
        if medical_features['size_consistency'] < 0.7:
            interpretation['clinical_notes'].append("Possible micrographia (small handwriting)")
        
        if medical_features['thickness_variation'] > 0.5:
            interpretation['clinical_notes'].append("Inconsistent line pressure (bradykinesia)")
        
        if medical_features['angular_consistency'] < 0.5:
            interpretation['clinical_notes'].append("Irregular movement patterns (rigidity)")
        
        # Confidence assessment
        if confidence < 0.7:
            interpretation['recommendations'].append("Low confidence - recommend additional tests")
        
        return interpretation
    
    def _print_prediction_results(self, result):
        """Print formatted prediction results"""
        print(f"\n🏥 Medical Analysis Results:")
        print(f"   📊 Prediction: {result['prediction']}")
        print(f"   🎯 Parkinson's Probability: {result['parkinson_probability']:.4f} ({result['parkinson_probability']*100:.2f}%)")
        print(f"   ✅ Healthy Probability: {result['healthy_probability']:.4f} ({result['healthy_probability']*100:.2f}%)")
        print(f"   🔒 Confidence: {result['confidence']:.4f}")
        
        print(f"\n🔬 Medical Features Analysis:")
        features = result['medical_features']
        print(f"   • Line Smoothness (Tremor): {features['line_smoothness']:.3f}")
        print(f"   • Size Consistency (Micrographia): {features['size_consistency']:.3f}")
        print(f"   • Thickness Variation (Bradykinesia): {features['thickness_variation']:.3f}")
        print(f"   • Angular Consistency (Rigidity): {features['angular_consistency']:.3f}")
        
        print(f"\n🩺 Clinical Interpretation:")
        clinical = result['clinical_interpretation']
        print(f"   • Risk Level: {clinical['risk_level']}")
        for note in clinical['clinical_notes']:
            print(f"   • {note}")
        
        print(f"\n💡 Recommendations:")
        for rec in clinical['recommendations']:
            print(f"   • {rec}")
    
    def _save_batch_results(self, results):
        """Save batch results to CSV"""
        # Flatten results for CSV
        csv_data = []
        for result in results:
            row = {
                'image_path': result['image_path'],
                'prediction': result['prediction'],
                'parkinson_probability': result['parkinson_probability'],
                'healthy_probability': result['healthy_probability'],
                'confidence': result['confidence'],
                'risk_level': result['clinical_interpretation']['risk_level'],
                'line_smoothness': result['medical_features']['line_smoothness'],
                'size_consistency': result['medical_features']['size_consistency'],
                'thickness_variation': result['medical_features']['thickness_variation'],
                'angular_consistency': result['medical_features']['angular_consistency']
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        filename = f'medical_parkinson_predictions_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filename, index=False)
        print(f"💾 Results saved to: {filename}")

def main():
    """Main interactive prediction function"""
    print("🏥 Medical Parkinson's Disease Prediction System")
    print("=" * 60)
    
    # Initialize predictor
    predictor = MedicalParkinsonPredictor()
    
    # Load trained model
    model_path = "medical_vit_parkinson_spiral.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using vision_transformer_parkinson.py")
        return
    
    predictor.load_model(model_path)
    
    while True:
        print("\n🎯 Choose prediction mode:")
        print("1. Single image prediction")
        print("2. Batch prediction")
        print("3. Analyze folder")
        print("4. Test with sample data")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip().replace('"', '')  # Remove quotes
            if os.path.exists(image_path):
                result = predictor.predict_single_image(image_path)
            else:
                print(f"❌ File not found: {image_path}")
        
        elif choice == '2':
            print("Enter image paths (one per line, empty line to finish):")
            image_paths = []
            while True:
                path = input().strip().replace('"', '')
                if not path:
                    break
                if os.path.exists(path):
                    image_paths.append(path)
                else:
                    print(f"❌ File not found: {path}")
            
            if image_paths:
                results = predictor.predict_batch(image_paths)
            else:
                print("❌ No valid image paths provided")
        
        elif choice == '3':
            folder_path = input("Enter folder path: ").strip().replace('"', '')
            if os.path.exists(folder_path):
                results = predictor.analyze_drawing_folder(folder_path)
            else:
                print(f"❌ Folder not found: {folder_path}")
        
        elif choice == '4':
            # Test with sample data from the dataset
            test_sample_data(predictor)
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")

def test_sample_data(predictor):
    """Test with some sample images from the dataset"""
    print("\n🔬 Testing with sample data from the dataset...")
    
    # Look for test images in the dataset
    dataset_path = "../data/Parkinson Dataset/dataset/spiral"
    
    sample_paths = []
    
    # Try to find some test images
    test_folders = [
        os.path.join(dataset_path, "testing", "healthy"),
        os.path.join(dataset_path, "testing", "parkinson"),
        os.path.join(dataset_path, "training", "healthy"),
        os.path.join(dataset_path, "training", "parkinson")
    ]
    
    for folder in test_folders:
        if os.path.exists(folder):
            import glob
            images = glob.glob(os.path.join(folder, "*.png"))
            if not images:
                images = glob.glob(os.path.join(folder, "*.jpg"))
            if images:
                # Take first 2 images from each category
                sample_paths.extend(images[:2])
    
    if sample_paths:
        print(f"📊 Found {len(sample_paths)} sample images")
        print("🔍 Analyzing sample images...")
        
        results = predictor.predict_batch(sample_paths[:8])  # Limit to 8 images
        
        # Show summary
        if results:
            print(f"\n📈 Sample Analysis Summary:")
            correct_predictions = 0
            total_predictions = len(results)
            
            for result in results:
                # Try to determine actual label from path
                actual_label = "Unknown"
                if "healthy" in result['image_path'].lower():
                    actual_label = "Healthy"
                elif "parkinson" in result['image_path'].lower():
                    actual_label = "Parkinson's Disease"
                
                predicted_label = result['prediction']
                is_correct = actual_label == predicted_label
                
                if is_correct and actual_label != "Unknown":
                    correct_predictions += 1
                
                print(f"   📄 {os.path.basename(result['image_path'])}")
                print(f"      Actual: {actual_label}, Predicted: {predicted_label}")
                print(f"      Confidence: {result['confidence']:.3f}, {'✅' if is_correct else '❌' if actual_label != 'Unknown' else '❓'}")
            
            if correct_predictions > 0:
                accuracy = correct_predictions / total_predictions * 100
                print(f"\n📊 Sample Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    else:
        print("❌ No sample images found in dataset")
        print("Please check if the dataset is in the correct location:")
        print("   ../data/Parkinson Dataset/dataset/spiral/")

def demo_prediction():
    """Demo function showing how to use the predictor"""
    print("🎯 Medical Parkinson's Predictor Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = MedicalParkinsonPredictor()
    
    # Load your trained model
    model_path = "medical_vit_parkinson_spiral.pth"  # Update with your model path
    if os.path.exists(model_path):
        predictor.load_model(model_path)
        
        # Example 1: Single image prediction
        print("\n📝 Example 1: Single Image Prediction")
        # Replace with actual image path
        # result = predictor.predict_single_image("path/to/your/spiral_drawing.png")
        
        # Example 2: Batch prediction
        print("\n📝 Example 2: Batch Prediction")
        # image_paths = ["image1.png", "image2.png", "image3.png"]
        # results = predictor.predict_batch(image_paths)
        
        # Example 3: Analyze entire folder
        print("\n📝 Example 3: Folder Analysis")
        # results = predictor.analyze_drawing_folder("path/to/drawings/folder")
        
        # Run the main interactive system
        main()
        
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure you have trained the model first!")

if __name__ == "__main__":
    demo_prediction()