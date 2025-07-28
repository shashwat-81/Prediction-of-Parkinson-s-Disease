import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image, ImageDraw, ImageFilter
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

class MedicalImagePreprocessor:
    """
    Medical-grade image preprocessing for Parkinson's drawing analysis
    """
    
    @staticmethod
    def extract_drawing_features(image_path):
        """Extract clinically relevant features from drawing images"""
        # Load image
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
            # Calculate line thickness variations
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
                    
                    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                    angles.append(angle)
                
                features['angular_consistency'] = 1 - (np.std(angles) / np.pi)
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
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Convert to grayscale if needed
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

class AdvancedParkinsonDataset(Dataset):
    """
    Advanced dataset with medical feature extraction for Parkinson's analysis
    """
    
    def __init__(self, data_paths, labels, transform=None, extract_medical_features=True):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform
        self.extract_medical_features = extract_medical_features
        self.medical_features = []
        
        if extract_medical_features:
            print("🔬 Extracting medical features from images...")
            for path in tqdm(data_paths, desc="Processing images"):
                features = MedicalImagePreprocessor.extract_drawing_features(path)
                self.medical_features.append(features if features else {
                    'line_smoothness': 0, 'size_consistency': 0, 
                    'thickness_variation': 0, 'angular_consistency': 0
                })
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        # Load and enhance image
        image = Image.open(self.data_paths[idx]).convert('RGB')
        image = MedicalImagePreprocessor.enhance_medical_features(image)
        
        label = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get medical features
        medical_features = torch.tensor([
            self.medical_features[idx]['line_smoothness'],
            self.medical_features[idx]['size_consistency'],
            self.medical_features[idx]['thickness_variation'],
            self.medical_features[idx]['angular_consistency']
        ], dtype=torch.float32) if self.extract_medical_features else torch.zeros(4)
        
        return image, label, medical_features, self.data_paths[idx]

class MedicalVisionTransformer(nn.Module):
    """
    Medical-grade Vision Transformer with clinical feature fusion
    """
    
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

class AdvancedParkinsonVIT:
    """
    Advanced Parkinson's disease detection using medical-informed Vision Transformer
    """
    
    def __init__(self, dataset_path="../data/Parkinson Dataset", image_size=224, batch_size=8, 
                 learning_rate=5e-5, num_epochs=100, dataset_type='spiral'):
        
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dataset_type = dataset_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🏥 Advanced Medical Vision Transformer for Parkinson's Detection")
        print(f"📱 Device: {self.device}")
        print(f"🧠 Medical feature fusion enabled")
        print(f"📊 Dataset: {dataset_type}")
        
        # Medical-grade transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=5),  # Minimal rotation to preserve medical features
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Subtle augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self.training_history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'train_auc': [], 'val_auc': []
        }
    
    def load_dataset(self):
        """Load dataset with medical feature extraction"""
        print("\n🔬 Loading dataset with medical analysis...")
        
        all_paths = []
        all_labels = []
        
        # Load data based on dataset type
        datasets_to_load = []
        if self.dataset_type == 'spiral':
            datasets_to_load = ['spiral']
        elif self.dataset_type == 'wave':
            datasets_to_load = ['wave']
        elif self.dataset_type == 'both':
            datasets_to_load = ['spiral', 'wave']
        
        for dataset_name in datasets_to_load:
            dataset_base_path = os.path.join(self.dataset_path, "dataset", dataset_name)
            
            for split in ['training', 'testing']:
                for class_name, label in [('healthy', 0), ('parkinson', 1)]:
                    class_path = os.path.join(dataset_base_path, split, class_name)
                    
                    if os.path.exists(class_path):
                        image_files = [f for f in os.listdir(class_path) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        
                        for image_file in image_files:
                            image_path = os.path.join(class_path, image_file)
                            all_paths.append(image_path)
                            all_labels.append(label)
        
        if not all_paths:
            raise ValueError("No images found!")
        
        print(f"📊 Dataset loaded: {len(all_paths)} images")
        print(f"   • Healthy: {all_labels.count(0)} images")
        print(f"   • Parkinson: {all_labels.count(1)} images")
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test (80% vs 20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
        )
        
        # Second split: train vs val (75% vs 25% of temp = 60% vs 20% of total)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )
        
        # Create datasets
        self.train_dataset = AdvancedParkinsonDataset(X_train, y_train, self.train_transform)
        self.val_dataset = AdvancedParkinsonDataset(X_val, y_val, self.val_transform)
        self.test_dataset = AdvancedParkinsonDataset(X_test, y_test, self.val_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, 
                                     shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, 
                                   shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, 
                                    shuffle=False, num_workers=0)
        
        print(f"✅ Data loaders created:")
        print(f"   • Train: {len(self.train_dataset)} samples, {len(self.train_loader)} batches")
        print(f"   • Validation: {len(self.val_dataset)} samples, {len(self.val_loader)} batches")
        print(f"   • Test: {len(self.test_dataset)} samples, {len(self.test_loader)} batches")
        
        return len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)
    
    def create_model(self):
        """Create the medical Vision Transformer model"""
        print("\n🧠 Creating Medical Vision Transformer...")
        
        self.model = MedicalVisionTransformer(num_classes=2, num_medical_features=4)
        self.model = self.model.to(self.device)
        
        # Loss function with class weighting for medical applications
        class_weights = torch.tensor([1.0, 1.2]).to(self.device)  # Slightly favor Parkinson's detection
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer with lower learning rate for medical stability
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, 
                                   weight_decay=0.01, betas=(0.9, 0.999))
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Model created:")
        print(f"   • Total parameters: {total_params:,}")
        print(f"   • Trainable parameters: {trainable_params:,}")
        print(f"   • Medical feature fusion: Enabled")
        print(f"   • Class-weighted loss: Enabled")
    
    def train_epoch(self):
        """Train for one epoch with medical metrics"""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        train_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, labels, medical_features, _ in train_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            medical_features = medical_features.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images, medical_features)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Get predictions for metrics - FIX: Use detach()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())
            all_probs.extend(probs[:, 1].cpu().detach().numpy())  # Parkinson's probability
            
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions) * 100
        epoch_auc = roc_auc_score(all_labels, all_probs)
        
        return epoch_loss, epoch_acc, epoch_auc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            val_bar = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for images, labels, medical_features, _ in val_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                medical_features = medical_features.to(self.device)
                
                outputs = self.model(images, medical_features)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # FIX: Use detach() - though not strictly needed in no_grad context
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                
                val_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions) * 100
        epoch_auc = roc_auc_score(all_labels, all_probs)
        
        return epoch_loss, epoch_acc, epoch_auc
    
    def train_model(self):
        """Train the complete model with medical validation"""
        print(f"\n🚀 Starting medical-grade training for {self.num_epochs} epochs...")
        
        best_val_auc = 0
        best_model_state = None
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train and validate
            train_loss, train_acc, train_auc = self.train_epoch()
            val_loss, val_acc, val_auc = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['train_auc'].append(train_auc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_auc'].append(val_auc)
            
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, AUC: {train_auc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model based on AUC (more important for medical applications)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"✅ New best validation AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{max_patience}")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\n⏹️ Early stopping after {epoch + 1} epochs")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n🎯 Training completed! Best validation AUC: {best_val_auc:.4f}")
        
        self.save_model()
        return best_val_auc
    
    def evaluate_model(self):
        """Medical-grade model evaluation"""
        print("\n🏥 Medical evaluation on test set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        all_paths = []
        
        with torch.no_grad():
            for images, labels, medical_features, paths in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                medical_features = medical_features.to(self.device)
                
                outputs = self.model(images, medical_features)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_paths.extend(paths)
        
        # Calculate comprehensive medical metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        
        # Calculate sensitivity and specificity (crucial for medical applications)
        cm = confusion_matrix(all_labels, all_predictions)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)  # True Positive Rate
        specificity = tn / (tn + fp)  # True Negative Rate
        
        print(f"\n🏥 Medical Test Results:")
        print(f"   📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   🎯 AUC-ROC: {auc:.4f}")
        print(f"   🔍 Sensitivity (Recall): {sensitivity:.4f} - Ability to detect Parkinson's")
        print(f"   ✅ Specificity: {specificity:.4f} - Ability to identify healthy patients")
        print(f"   ⚖️ Precision: {precision:.4f}")
        print(f"   📈 F1-Score: {f1:.4f}")
        
        # Clinical interpretation
        print(f"\n🩺 Clinical Interpretation:")
        if sensitivity >= 0.85 and specificity >= 0.80:
            print("   ✅ Excellent diagnostic performance suitable for clinical screening")
        elif sensitivity >= 0.75 and specificity >= 0.70:
            print("   ⚠️ Good performance, suitable for assisted diagnosis")
        else:
            print("   ❌ Performance needs improvement for clinical use")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_probs])
        
        plt.figure(figsize=(12, 5))
        
        # ROC Curve
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve - Medical ViT')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Confusion Matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Healthy', 'Parkinson'], 
                   yticklabels=['Healthy', 'Parkinson'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'medical_evaluation_{self.dataset_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save comprehensive results
        results = {
            'accuracy': accuracy,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probs,
            'file_paths': all_paths
        }
        
        joblib.dump(results, f'medical_vit_results_{self.dataset_type}.pkl')
        print(f"✅ Results saved to: medical_vit_results_{self.dataset_type}.pkl")
        
        return results
    
    def save_model(self):
        """Save the medical model"""
        model_path = f'medical_vit_parkinson_{self.dataset_type}.pth'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': 'MedicalVisionTransformer',
            'dataset_type': self.dataset_type,
            'training_history': self.training_history,
            'medical_features': ['line_smoothness', 'size_consistency', 'thickness_variation', 'angular_consistency']
        }, model_path)
        
        print(f"💾 Medical model saved to: {model_path}")

def main():
    """Main function for medical-grade Parkinson's detection"""
    print("🏥 Medical Vision Transformer for Parkinson's Disease Detection")
    print("=" * 70)
    print("Based on clinical understanding of motor symptoms:")
    print("• Tremor analysis through line smoothness")
    print("• Micrographia detection via size consistency")
    print("• Bradykinesia assessment through thickness variation")
    print("• Rigidity evaluation via angular consistency")
    print("=" * 70)
    
    try:
        # Initialize medical model
        medical_vit = AdvancedParkinsonVIT(
            dataset_path="../data/Parkinson Dataset",
            image_size=224,
            batch_size=8,  # Smaller batch for better gradient stability
            learning_rate=5e-5,  # Lower LR for medical precision
            num_epochs=100,
            dataset_type='spiral'
        )
        
        # Load and process data
        train_size, val_size, test_size = medical_vit.load_dataset()
        
        # Create medical model
        medical_vit.create_model()
        
        # Train with medical validation
        best_auc = medical_vit.train_model()
        
        # Medical evaluation
        results = medical_vit.evaluate_model()
        
        print(f"\n🎉 Medical training completed!")
        print(f"🏥 Final Medical Metrics:")
        print(f"   • Best Validation AUC: {best_auc:.4f}")
        print(f"   • Test AUC: {results['auc']:.4f}")
        print(f"   • Sensitivity: {results['sensitivity']:.4f}")
        print(f"   • Specificity: {results['specificity']:.4f}")
        
    except Exception as e:
        print(f"❌ Error in medical training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
