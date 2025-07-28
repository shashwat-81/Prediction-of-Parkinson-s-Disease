import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vit_b_16
import torchvision.transforms as transforms
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class VisionTransformerPredictor:
    """
    Vision Transformer predictor for Parkinson's disease detection from drawing images
    """
    
    def __init__(self, model_path="vision_transformer_parkinson_spiral.pth", device=None):
        """
        Initialize the predictor with a trained Vision Transformer model
        
        Args:
            model_path (str): Path to the trained model file
            device: Device to run inference on (cuda/cpu)
        """
        self.model_path = model_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = ['Healthy', 'Parkinson']
        self.image_size = 224
        
        print(f"🎯 Vision Transformer Parkinson's Predictor")
        print(f"📱 Using device: {self.device}")
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load the trained model
        self.load_model()
    
    def load_model(self):
        """Load the trained Vision Transformer model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"📂 Loading model from: {self.model_path}")
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model information
            self.image_size = checkpoint.get('image_size', 224)
            self.class_names = checkpoint.get('class_names', ['Healthy', 'Parkinson'])
            dataset_type = checkpoint.get('dataset_type', 'spiral')
            
            print(f"✓ Model info: Image size: {self.image_size}, Dataset: {dataset_type}")
            
            # Create model architecture
            self.model = vit_b_16(weights=None)  # Don't load pre-trained weights
            
            # Modify classifier head to match saved model
            num_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 2)  # 2 classes: healthy, parkinson
            )
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def predict_single_image(self, image_path, return_attention=False):
        """
        Predict Parkinson's disease for a single image
        
        Args:
            image_path (str): Path to the image file
            return_attention (bool): Whether to return attention weights
            
        Returns:
            dict: Prediction results
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        print(f"🔍 Analyzing image: {os.path.basename(image_path)}")
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        if image_tensor is None:
            return {"error": "Failed to preprocess image"}
        
        try:
            with torch.no_grad():
                # Get model predictions
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                
                # Extract probabilities
                healthy_prob = probabilities[0][0].item()
                parkinson_prob = probabilities[0][1].item()
                confidence = probabilities[0][predicted_class_idx].item()
                
                # Determine risk level
                risk_level = self.get_risk_level(parkinson_prob)
                
                # Prepare results
                result = {
                    "filename": os.path.basename(image_path),
                    "prediction": self.class_names[predicted_class_idx],
                    "confidence": confidence,
                    "healthy_probability": healthy_prob,
                    "parkinson_probability": parkinson_prob,
                    "risk_level": risk_level,
                    "recommendation": self.get_recommendation(parkinson_prob),
                    "model_type": "Vision Transformer"
                }
                
                return result
                
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
    
    def predict_multiple_images(self, folder_path, output_csv=None):
        """
        Predict Parkinson's disease for multiple images in a folder
        
        Args:
            folder_path (str): Path to folder containing images
            output_csv (str): Optional path to save results as CSV
            
        Returns:
            list: List of prediction results
        """
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return []
        
        # Find image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print("No image files found in the folder")
            return []
        
        print(f"📁 Found {len(image_files)} images to analyze")
        
        results = []
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(folder_path, filename)
            print(f"\n[{i}/{len(image_files)}] Processing: {filename}")
            
            result = self.predict_single_image(image_path)
            if "error" not in result:
                results.append(result)
                
                # Print summary
                print(f"  → Prediction: {result['prediction']}")
                print(f"  → Confidence: {result['confidence']:.3f}")
                print(f"  → Parkinson's probability: {result['parkinson_probability']:.3f}")
                print(f"  → Risk level: {result['risk_level']}")
            else:
                print(f"  → Error: {result['error']}")
        
        # Save results to CSV if requested
        if output_csv and results:
            self.save_results_to_csv(results, output_csv)
        
        return results
    
    def save_results_to_csv(self, results, output_path):
        """Save prediction results to CSV file"""
        try:
            import pandas as pd
            
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"\n✅ Results saved to: {output_path}")
            
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
    
    def get_recommendation(self, parkinson_prob):
        """Get clinical recommendation based on prediction"""
        if parkinson_prob < 0.3:
            return "Low concern. Continue regular monitoring."
        elif parkinson_prob < 0.6:
            return "Moderate concern. Consider consultation with neurologist."
        elif parkinson_prob < 0.8:
            return "High concern. Recommend neurological evaluation."
        else:
            return "Very high concern. Urgent neurological consultation recommended."
    
    def visualize_prediction(self, image_path, save_plot=True):
        """
        Visualize the prediction with the original image
        
        Args:
            image_path (str): Path to the image file
            save_plot (bool): Whether to save the visualization
        """
        # Get prediction
        result = self.predict_single_image(image_path)
        
        if "error" in result:
            print(f"Cannot visualize: {result['error']}")
            return
        
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display original image
        ax1.imshow(original_image)
        ax1.set_title(f"Original Image: {result['filename']}", fontsize=14)
        ax1.axis('off')
        
        # Display prediction results
        colors = ['green' if result['prediction'] == 'Healthy' else 'red']
        probabilities = [result['healthy_probability'], result['parkinson_probability']]
        labels = ['Healthy', 'Parkinson']
        
        bars = ax2.bar(labels, probabilities, color=['lightgreen', 'lightcoral'])
        ax2.set_title('Prediction Probabilities', fontsize=14)
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        
        # Add probability labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add prediction info as text
        info_text = f"""Prediction: {result['prediction']}
Confidence: {result['confidence']:.3f}
Risk Level: {result['risk_level']}

Recommendation:
{result['recommendation']}"""
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_plot:
            plot_filename = f"prediction_{os.path.splitext(result['filename'])[0]}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to: {plot_filename}")
        
        plt.show()
    
    def print_detailed_report(self, result):
        """Print a detailed prediction report"""
        print("\n" + "="*60)
        print("VISION TRANSFORMER PARKINSON'S PREDICTION REPORT")
        print("="*60)
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
            return
        
        print(f"📁 File: {result['filename']}")
        print(f"🤖 Model: {result['model_type']}")
        
        print(f"\n🎯 PREDICTION RESULTS")
        print(f"   Predicted Class: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Healthy Probability: {result['healthy_probability']:.4f}")
        print(f"   Parkinson's Probability: {result['parkinson_probability']:.4f}")
        print(f"   Risk Level: {result['risk_level']}")
        
        print(f"\n💡 CLINICAL RECOMMENDATION")
        print(f"   {result['recommendation']}")
        
        print("="*60)

def compare_with_audio_model(image_path, audio_model_path="wav2vec_best_model.pkl"):
    """
    Compare Vision Transformer results with audio-based model if available
    
    Args:
        image_path (str): Path to the image file
        audio_model_path (str): Path to the audio model
    """
    # Vision Transformer prediction
    vit_predictor = VisionTransformerPredictor()
    vit_result = vit_predictor.predict_single_image(image_path)
    
    print("🔄 COMPARISON: Vision Transformer vs Audio Model")
    print("="*60)
    
    print("🖼️ Vision Transformer (Drawing Analysis):")
    if "error" not in vit_result:
        print(f"   Prediction: {vit_result['prediction']}")
        print(f"   Parkinson's Probability: {vit_result['parkinson_probability']:.3f}")
        print(f"   Risk Level: {vit_result['risk_level']}")
    else:
        print(f"   Error: {vit_result['error']}")
    
    # Note about audio model
    print("\n🎤 Audio Model (Voice Analysis):")
    print("   Note: Audio model requires voice recordings (.wav files)")
    print("   Current input is an image file, so audio analysis is not applicable.")
    
    print("\n💡 RECOMMENDATION:")
    print("   For comprehensive diagnosis, consider using both:")
    print("   • Drawing analysis (Vision Transformer) - for motor symptoms")
    print("   • Voice analysis (Audio model) - for speech symptoms")

def main():
    """Main function for command-line usage"""
    print("🎯 Vision Transformer Parkinson's Disease Prediction System")
    print("=" * 60)
    
    # Initialize predictor
    try:
        predictor = VisionTransformerPredictor()
        print("✅ Vision Transformer model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return
    
    while True:
        print("\nOptions:")
        print("1. Predict single image")
        print("2. Predict multiple images from folder")
        print("3. Visualize prediction")
        print("4. Compare with audio model")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            image_path = input("Enter path to image file: ").strip().strip('"')
            if image_path:
                result = predictor.predict_single_image(image_path)
                predictor.print_detailed_report(result)
        
        elif choice == "2":
            folder_path = input("Enter path to folder containing images: ").strip().strip('"')
            save_csv = input("Save results to CSV? (y/n): ").strip().lower() == 'y'
            
            if folder_path:
                output_csv = None
                if save_csv:
                    output_csv = input("Enter CSV filename (or press Enter for default): ").strip()
                    if not output_csv:
                        output_csv = "vit_prediction_results.csv"
                
                results = predictor.predict_multiple_images(folder_path, output_csv)
                
                if results:
                    print(f"\n📊 SUMMARY OF {len(results)} PREDICTIONS")
                    print("-" * 40)
                    
                    risk_counts = {}
                    for result in results:
                        risk = result.get('risk_level', 'Unknown')
                        risk_counts[risk] = risk_counts.get(risk, 0) + 1
                    
                    for risk, count in risk_counts.items():
                        print(f"{risk}: {count} images")
        
        elif choice == "3":
            image_path = input("Enter path to image file for visualization: ").strip().strip('"')
            if image_path:
                predictor.visualize_prediction(image_path)
        
        elif choice == "4":
            image_path = input("Enter path to image file for comparison: ").strip().strip('"')
            if image_path:
                compare_with_audio_model(image_path)
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
