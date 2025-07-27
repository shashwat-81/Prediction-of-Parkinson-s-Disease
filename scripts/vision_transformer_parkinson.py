import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import joblib
import warnings
from torch.cuda.amp import GradScaler, autocast
warnings.filterwarnings('ignore')

class ParkinsonDrawingDataset(Dataset):
    """
    Custom dataset for Parkinson's disease drawing images
    """
    
    def __init__(self, data_paths, labels, transform=None, dataset_type='spiral'):
        """
        Initialize the dataset
        
        Args:
            data_paths (list): List of image file paths
            labels (list): List of corresponding labels (0: healthy, 1: parkinson)
            transform: Transformations to apply to images
            dataset_type (str): Type of dataset ('spiral' or 'wave')
        """
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform
        self.dataset_type = dataset_type
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        try:
            # Load image
            image_path = self.data_paths[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            return image, label, image_path
        except Exception as e:
            print(f"Error loading image {self.data_paths[idx]}: {e}")
            # Return a black image as fallback
            if self.transform:
                black_image = Image.new('RGB', (224, 224), (0, 0, 0))
                image = self.transform(black_image)
            else:
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            return image, self.labels[idx], self.data_paths[idx]

class VisionTransformerParkinson:
    """
    Vision Transformer model for Parkinson's disease prediction using drawing images
    """
    
    def __init__(self, dataset_path="data/Parkinson Dataset", image_size=224, batch_size=8, 
                 learning_rate=3e-5, num_epochs=100, dataset_type='spiral'):
        """
        Initialize the Vision Transformer model
        
        Args:
            dataset_path (str): Path to the dataset
            image_size (int): Size to resize images to
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            num_epochs (int): Number of training epochs
            dataset_type (str): Type of dataset to use ('spiral', 'wave', or 'both')
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dataset_type = dataset_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🎯 Initializing Vision Transformer for Parkinson's Detection")
        print(f"📱 Using device: {self.device}")
        print(f"🖼️ Image size: {image_size}x{image_size}")
        print(f"📊 Dataset type: {dataset_type}")
        
        # Initialize model components
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.scaler = GradScaler()  # For mixed precision training
        
        # Define transformations with simplified augmentation for stability
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_dataset(self):
        """Load and prepare the dataset"""
        print("\n📂 Loading dataset...")
        
        all_paths = []
        all_labels = []
        
        # Define dataset types to load
        datasets_to_load = []
        if self.dataset_type == 'spiral':
            datasets_to_load = ['spiral']
        elif self.dataset_type == 'wave':
            datasets_to_load = ['wave']
        elif self.dataset_type == 'both':
            datasets_to_load = ['spiral', 'wave']
        
        for dataset_name in datasets_to_load:
            dataset_base_path = os.path.join(self.dataset_path, "dataset", dataset_name)
            print(f"🔍 Looking for dataset at: {dataset_base_path}")
            
            # Load training data
            for split in ['training', 'testing']:
                for class_name, label in [('healthy', 0), ('parkinson', 1)]:
                    class_path = os.path.join(dataset_base_path, split, class_name)
                    print(f"🔍 Checking path: {class_path}")
                    print(f"🔍 Path exists: {os.path.exists(class_path)}")
                    
                    if os.path.exists(class_path):
                        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        
                        for image_file in image_files:
                            image_path = os.path.join(class_path, image_file)
                            all_paths.append(image_path)
                            all_labels.append(label)
                        
                        print(f"✓ Loaded {len(image_files)} {class_name} images from {dataset_name}/{split}")
        
        if not all_paths:
            raise ValueError("No images found in the dataset!")
        
        print(f"\n📊 Dataset Summary:")
        print(f"   • Total images: {len(all_paths)}")
        print(f"   • Healthy images: {all_labels.count(0)}")
        print(f"   • Parkinson images: {all_labels.count(1)}")
        print(f"   • Class balance: {all_labels.count(0)/len(all_labels):.2f} healthy, {all_labels.count(1)/len(all_labels):.2f} parkinson")
        
        # Create full dataset
        full_dataset = ParkinsonDrawingDataset(all_paths, all_labels, transform=None, dataset_type=self.dataset_type)
        
        # Split dataset: 70% train, 15% validation, 15% test
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        # Create indices for stratified split
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create datasets with appropriate transforms
        train_paths = [all_paths[i] for i in train_indices]
        train_labels = [all_labels[i] for i in train_indices]
        train_dataset = ParkinsonDrawingDataset(train_paths, train_labels, transform=self.train_transform)
        
        val_paths = [all_paths[i] for i in val_indices]
        val_labels = [all_labels[i] for i in val_indices]
        val_dataset = ParkinsonDrawingDataset(val_paths, val_labels, transform=self.val_transform)
        
        test_paths = [all_paths[i] for i in test_indices]
        test_labels = [all_labels[i] for i in test_indices]
        test_dataset = ParkinsonDrawingDataset(test_paths, test_labels, transform=self.val_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        print(f"\n📦 Data Loaders Created:")
        print(f"   • Training batches: {len(self.train_loader)} (batch size: {self.batch_size})")
        print(f"   • Validation batches: {len(self.val_loader)}")
        print(f"   • Test batches: {len(self.test_loader)}")
        
        return len(train_dataset), len(val_dataset), len(test_dataset)
    
    def create_model(self):
        """Create and initialize the Vision Transformer model"""
        print("\n🤖 Creating Vision Transformer model...")
        
        # Load pre-trained Vision Transformer
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Freeze early layers for better fine-tuning
        for param in self.model.encoder.layers[:-3].parameters():
            param.requires_grad = False
        
        # Modify the classifier head for binary classification (simplified)
        num_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_features, 2)  # 2 classes: healthy, parkinson
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Vision Transformer model created")
        print(f"   • Total parameters: {total_params:,}")
        print(f"   • Trainable parameters: {trainable_params:,}")
        print(f"   • Frozen parameters: {total_params - trainable_params:,}")
        
        # Define loss function and optimizer (simplified)
        self.criterion = nn.CrossEntropyLoss()
        
        # Simple optimizer for all parameters
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        # Simple scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.7)
        
        print(f"✓ Optimizer: AdamW (lr={self.learning_rate})")
        print(f"✓ Loss function: CrossEntropyLoss")
        print(f"✓ Scheduler: StepLR")
    
    def train_epoch(self):
        """Train the model for one epoch with mixed precision"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        train_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, labels, _) in enumerate(train_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate the model for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            val_bar = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for images, labels, _ in val_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Update progress bar
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def train_model(self):
        """Train the complete model"""
        print(f"\n🚀 Starting training for {self.num_epochs} epochs...")
        
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        max_patience = 15  # Reasonable patience
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 40)
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"✓ New best validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{max_patience}")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\n⏹️ Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f'checkpoint_epoch_{epoch+1}_{self.dataset_type}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'training_history': self.training_history
                }, checkpoint_path)
                print(f"📁 Checkpoint saved: {checkpoint_path}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
        # Save the trained model
        self.save_model()
        
        return best_val_acc
    
    def evaluate_model(self):
        """Evaluate the model on test set"""
        print("\n🧪 Evaluating model on test set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            test_bar = tqdm(self.test_loader, desc="Testing")
            
            for images, labels, _ in test_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        print(f"\n📊 Test Results:")
        print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   • Precision: {precision:.4f}")
        print(f"   • Recall: {recall:.4f}")
        print(f"   • F1-Score: {f1:.4f}")
        
        # Detailed classification report
        class_names = ['Healthy', 'Parkinson']
        report = classification_report(all_labels, all_predictions, target_names=class_names)
        print(f"\n📋 Detailed Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Vision Transformer')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_vit_{self.dataset_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save evaluation results
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        joblib.dump(evaluation_results, f'vit_evaluation_results_{self.dataset_type}.pkl')
        print(f"✓ Evaluation results saved to: vit_evaluation_results_{self.dataset_type}.pkl")
        
        return evaluation_results
    
    def plot_training_history(self):
        """Plot training history"""
        print("\n📈 Plotting training history...")
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.training_history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_history_vit_{self.dataset_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Training history plot saved to: training_history_vit_{self.dataset_type}.png")
    
    def save_model(self):
        """Save the trained model"""
        model_save_path = f'vision_transformer_parkinson_{self.dataset_type}.pth'
        
        # Save complete model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': 'vit_b_16',
            'num_classes': 2,
            'image_size': self.image_size,
            'dataset_type': self.dataset_type,
            'training_history': self.training_history,
            'class_names': ['Healthy', 'Parkinson']
        }, model_save_path)
        
        print(f"✓ Model saved to: {model_save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint to resume training"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        self.training_history = checkpoint['training_history']
        
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        
        print(f"✓ Checkpoint loaded successfully!")
        print(f"   • Resuming from epoch: {start_epoch}")
        print(f"   • Best validation accuracy so far: {best_val_acc:.2f}%")
        
        return start_epoch, best_val_acc
    
    def train_with_resume(self, checkpoint_path=None):
        """Train model with option to resume from checkpoint"""
        start_epoch = 0
        best_val_acc = 0
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            start_epoch, best_val_acc = self.load_checkpoint(checkpoint_path)
        
        print(f"\n🚀 Starting training from epoch {start_epoch + 1} to {self.num_epochs}...")
        
        best_model_state = None
        patience_counter = 0
        max_patience = 20
        improvement_threshold = 0.5
        
        for epoch in range(start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 40)
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            if epoch >= len(self.training_history['train_loss']):
                self.training_history['train_loss'].append(train_loss)
                self.training_history['train_acc'].append(train_acc)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_acc'].append(val_acc)
            else:
                self.training_history['train_loss'][epoch] = train_loss
                self.training_history['train_acc'][epoch] = train_acc
                self.training_history['val_loss'][epoch] = val_loss
                self.training_history['val_acc'][epoch] = val_acc
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.8f}")
            
            # Save best model
            if val_acc > best_val_acc + improvement_threshold:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"✓ New best validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{max_patience}")
            
            # Early stopping
            if patience_counter >= max_patience and epoch > 30:
                print(f"\n⏹️ Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_save_path = f'checkpoint_epoch_{epoch+1}_{self.dataset_type}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'training_history': self.training_history
                }, checkpoint_save_path)
                print(f"📁 Checkpoint saved: {checkpoint_save_path}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
        # Save the trained model
        self.save_model()
        
        return best_val_acc
    
    def predict_single_image(self, image_path):
        """Predict on a single image"""
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Get results
        pred_class = predicted.item()
        pred_prob = probabilities[0][pred_class].item()
        healthy_prob = probabilities[0][0].item()
        parkinson_prob = probabilities[0][1].item()
        
        class_names = ['Healthy', 'Parkinson']
        
        result = {
            'prediction': class_names[pred_class],
            'confidence': pred_prob,
            'healthy_probability': healthy_prob,
            'parkinson_probability': parkinson_prob,
            'risk_level': 'Low' if parkinson_prob < 0.3 else 'Moderate' if parkinson_prob < 0.7 else 'High'
        }
        
        return result

def main():
    """Main function to train and evaluate the Vision Transformer model"""
    print("🎯 Vision Transformer for Parkinson's Disease Detection")
    print("=" * 60)
    
    # Initialize the model with stable parameters
    vit_model = VisionTransformerParkinson(
        dataset_path="data/Parkinson Dataset",
        image_size=224,
        batch_size=4,  # Even smaller batch size for stability
        learning_rate=1e-4,  # Standard learning rate
        num_epochs=30,  # Fewer epochs for testing
        dataset_type='spiral'  # Change to 'wave' or 'both' as needed
    )
    
    try:
        # Load dataset
        train_size, val_size, test_size = vit_model.load_dataset()
        
        # Create model
        vit_model.create_model()
        
        # Train model
        best_acc = vit_model.train_model()
        
        # Plot training history
        vit_model.plot_training_history()
        
        # Evaluate model
        evaluation_results = vit_model.evaluate_model()
        
        print(f"\n🎉 Training and evaluation completed!")
        print(f"📊 Final Results Summary:")
        print(f"   • Best Validation Accuracy: {best_acc:.2f}%")
        print(f"   • Test Accuracy: {evaluation_results['accuracy']*100:.2f}%")
        print(f"   • Test F1-Score: {evaluation_results['f1_score']:.4f}")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
