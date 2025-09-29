import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ParkinsonDatasetAnalyzer:
    """
    Comprehensive analyzer for Parkinson's disease drawing dataset
    """
    
    def __init__(self, dataset_path="../data/Parkinson Dataset"):
        """
        Initialize the dataset analyzer
        
        Args:
            dataset_path (str): Path to the Parkinson dataset
        """
        self.dataset_path = dataset_path
        self.analysis_results = {}
        
    def analyze_dataset_structure(self):
        """Analyze the overall structure of the dataset"""
        print("🔍 PARKINSON'S DATASET STRUCTURE ANALYSIS")
        print("=" * 60)
        
        # Check main dataset structure
        spiral_path = os.path.join(self.dataset_path, "dataset", "spiral")
        wave_path = os.path.join(self.dataset_path, "dataset", "wave")
        digital_path = os.path.join(self.dataset_path, "Digital Drawings spiral")
        
        structure_info = {
            "spiral_exists": os.path.exists(spiral_path),
            "wave_exists": os.path.exists(wave_path),
            "digital_exists": os.path.exists(digital_path)
        }
        
        print(f"📁 Dataset Components:")
        print(f"   • Spiral drawings: {'✓' if structure_info['spiral_exists'] else '✗'}")
        print(f"   • Wave drawings: {'✓' if structure_info['wave_exists'] else '✗'}")
        print(f"   • Digital drawings: {'✓' if structure_info['digital_exists'] else '✗'}")
        
        return structure_info
    
    def analyze_spiral_dataset(self):
        """Analyze spiral drawing dataset"""
        print("\n🌀 SPIRAL DATASET ANALYSIS")
        print("-" * 40)
        
        spiral_path = os.path.join(self.dataset_path, "dataset", "spiral")
        
        # Training data analysis
        train_healthy_path = os.path.join(spiral_path, "training", "healthy")
        train_parkinson_path = os.path.join(spiral_path, "training", "parkinson")
        test_healthy_path = os.path.join(spiral_path, "testing", "healthy")
        test_parkinson_path = os.path.join(spiral_path, "testing", "parkinson")
        
        spiral_stats = {}
        
        if os.path.exists(train_healthy_path):
            train_healthy_files = [f for f in os.listdir(train_healthy_path) if f.endswith('.png')]
            spiral_stats['train_healthy'] = len(train_healthy_files)
        else:
            spiral_stats['train_healthy'] = 0
            
        if os.path.exists(train_parkinson_path):
            train_parkinson_files = [f for f in os.listdir(train_parkinson_path) if f.endswith('.png')]
            spiral_stats['train_parkinson'] = len(train_parkinson_files)
        else:
            spiral_stats['train_parkinson'] = 0
            
        if os.path.exists(test_healthy_path):
            test_healthy_files = [f for f in os.listdir(test_healthy_path) if f.endswith('.png')]
            spiral_stats['test_healthy'] = len(test_healthy_files)
        else:
            spiral_stats['test_healthy'] = 0
            
        if os.path.exists(test_parkinson_path):
            test_parkinson_files = [f for f in os.listdir(test_parkinson_path) if f.endswith('.png')]
            spiral_stats['test_parkinson'] = len(test_parkinson_files)
        else:
            spiral_stats['test_parkinson'] = 0
        
        print(f"📊 Training Data:")
        print(f"   • Healthy: {spiral_stats['train_healthy']} images")
        print(f"   • Parkinson: {spiral_stats['train_parkinson']} images")
        print(f"   • Total Training: {spiral_stats['train_healthy'] + spiral_stats['train_parkinson']}")
        
        print(f"📊 Testing Data:")
        print(f"   • Healthy: {spiral_stats['test_healthy']} images")
        print(f"   • Parkinson: {spiral_stats['test_parkinson']} images")
        print(f"   • Total Testing: {spiral_stats['test_healthy'] + spiral_stats['test_parkinson']}")
        
        # Calculate class distribution
        total_healthy = spiral_stats['train_healthy'] + spiral_stats['test_healthy']
        total_parkinson = spiral_stats['train_parkinson'] + spiral_stats['test_parkinson']
        total_images = total_healthy + total_parkinson
        
        if total_images > 0:
            healthy_ratio = (total_healthy / total_images) * 100
            parkinson_ratio = (total_parkinson / total_images) * 100
            
            print(f"\n📈 Class Distribution:")
            print(f"   • Healthy: {healthy_ratio:.1f}% ({total_healthy} images)")
            print(f"   • Parkinson: {parkinson_ratio:.1f}% ({total_parkinson} images)")
            print(f"   • Balance Ratio: {min(healthy_ratio, parkinson_ratio) / max(healthy_ratio, parkinson_ratio):.2f}")
        
        self.analysis_results['spiral'] = spiral_stats
        return spiral_stats
    
    def analyze_wave_dataset(self):
        """Analyze wave drawing dataset"""
        print("\n🌊 WAVE DATASET ANALYSIS")
        print("-" * 40)
        
        wave_path = os.path.join(self.dataset_path, "dataset", "wave")
        
        # Training data analysis
        train_healthy_path = os.path.join(wave_path, "training", "healthy")
        train_parkinson_path = os.path.join(wave_path, "training", "parkinson")
        test_healthy_path = os.path.join(wave_path, "testing", "healthy")
        test_parkinson_path = os.path.join(wave_path, "testing", "parkinson")
        
        wave_stats = {}
        
        for split, path_type in [('train', 'training'), ('test', 'testing')]:
            for class_name in ['healthy', 'parkinson']:
                path = os.path.join(wave_path, path_type, class_name)
                if os.path.exists(path):
                    files = [f for f in os.listdir(path) if f.endswith('.png')]
                    wave_stats[f'{split}_{class_name}'] = len(files)
                else:
                    wave_stats[f'{split}_{class_name}'] = 0
        
        print(f"📊 Training Data:")
        print(f"   • Healthy: {wave_stats.get('train_healthy', 0)} images")
        print(f"   • Parkinson: {wave_stats.get('train_parkinson', 0)} images")
        print(f"   • Total Training: {wave_stats.get('train_healthy', 0) + wave_stats.get('train_parkinson', 0)}")
        
        print(f"📊 Testing Data:")
        print(f"   • Healthy: {wave_stats.get('test_healthy', 0)} images")
        print(f"   • Parkinson: {wave_stats.get('test_parkinson', 0)} images")
        print(f"   • Total Testing: {wave_stats.get('test_healthy', 0) + wave_stats.get('test_parkinson', 0)}")
        
        self.analysis_results['wave'] = wave_stats
        return wave_stats
    
    def analyze_image_properties(self, sample_size=10):
        """Analyze image properties like dimensions, color channels, etc."""
        print("\n🖼️ IMAGE PROPERTIES ANALYSIS")
        print("-" * 40)
        
        image_properties = {
            'spiral': {'sizes': [], 'channels': [], 'modes': []},
            'wave': {'sizes': [], 'channels': [], 'modes': []}
        }
        
        # Analyze spiral images
        spiral_path = os.path.join(self.dataset_path, "dataset", "spiral", "training")
        for class_name in ['healthy', 'parkinson']:
            class_path = os.path.join(spiral_path, class_name)
            if os.path.exists(class_path):
                files = [f for f in os.listdir(class_path) if f.endswith('.png')][:sample_size]
                for file in files:
                    try:
                        img_path = os.path.join(class_path, file)
                        img = Image.open(img_path)
                        image_properties['spiral']['sizes'].append(img.size)
                        image_properties['spiral']['channels'].append(len(img.getbands()))
                        image_properties['spiral']['modes'].append(img.mode)
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
        
        # Analyze wave images
        wave_path = os.path.join(self.dataset_path, "dataset", "wave", "training")
        for class_name in ['healthy', 'parkinson']:
            class_path = os.path.join(wave_path, class_name)
            if os.path.exists(class_path):
                files = [f for f in os.listdir(class_path) if f.endswith('.png')][:sample_size]
                for file in files:
                    try:
                        img_path = os.path.join(class_path, file)
                        img = Image.open(img_path)
                        image_properties['wave']['sizes'].append(img.size)
                        image_properties['wave']['channels'].append(len(img.getbands()))
                        image_properties['wave']['modes'].append(img.mode)
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
        
        # Print analysis results
        for dataset_type in ['spiral', 'wave']:
            if image_properties[dataset_type]['sizes']:
                sizes = image_properties[dataset_type]['sizes']
                channels = image_properties[dataset_type]['channels']
                modes = image_properties[dataset_type]['modes']
                
                print(f"\n{dataset_type.upper()} Images:")
                print(f"   • Sample size: {len(sizes)} images")
                print(f"   • Image dimensions: {list(set(sizes))}")
                print(f"   • Color channels: {list(set(channels))}")
                print(f"   • Image modes: {list(set(modes))}")
                
                if sizes:
                    avg_width = np.mean([s[0] for s in sizes])
                    avg_height = np.mean([s[1] for s in sizes])
                    print(f"   • Average size: {avg_width:.0f} x {avg_height:.0f}")
        
        self.analysis_results['image_properties'] = image_properties
        return image_properties
    
    def create_visualization(self, save_path="dataset_analysis.png"):
        """Create visualizations of the dataset analysis"""
        print(f"\n📊 Creating visualization...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parkinson\'s Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Dataset distribution pie chart
        if 'spiral' in self.analysis_results:
            spiral_stats = self.analysis_results['spiral']
            total_healthy = spiral_stats.get('train_healthy', 0) + spiral_stats.get('test_healthy', 0)
            total_parkinson = spiral_stats.get('train_parkinson', 0) + spiral_stats.get('test_parkinson', 0)
            
            if total_healthy + total_parkinson > 0:
                labels = ['Healthy', 'Parkinson']
                sizes = [total_healthy, total_parkinson]
                colors = ['#66b3ff', '#ff9999']
                
                axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[0, 0].set_title('Spiral Dataset - Class Distribution')
        
        # 2. Train vs Test distribution
        if 'spiral' in self.analysis_results:
            spiral_stats = self.analysis_results['spiral']
            categories = ['Train Healthy', 'Train Parkinson', 'Test Healthy', 'Test Parkinson']
            values = [
                spiral_stats.get('train_healthy', 0),
                spiral_stats.get('train_parkinson', 0),
                spiral_stats.get('test_healthy', 0),
                spiral_stats.get('test_parkinson', 0)
            ]
            
            colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99']
            bars = axes[0, 1].bar(range(len(categories)), values, color=colors)
            axes[0, 1].set_title('Spiral Dataset - Train/Test Split')
            axes[0, 1].set_xticks(range(len(categories)))
            axes[0, 1].set_xticklabels(categories, rotation=45, ha='right')
            axes[0, 1].set_ylabel('Number of Images')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               str(value), ha='center', va='bottom')
        
        # 3. Wave dataset if available
        if 'wave' in self.analysis_results:
            wave_stats = self.analysis_results['wave']
            total_healthy = wave_stats.get('train_healthy', 0) + wave_stats.get('test_healthy', 0)
            total_parkinson = wave_stats.get('train_parkinson', 0) + wave_stats.get('test_parkinson', 0)
            
            if total_healthy + total_parkinson > 0:
                labels = ['Healthy', 'Parkinson']
                sizes = [total_healthy, total_parkinson]
                colors = ['#66b3ff', '#ff9999']
                
                axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title('Wave Dataset - Class Distribution')
        
        # 4. Combined statistics
        if 'spiral' in self.analysis_results and 'wave' in self.analysis_results:
            datasets = ['Spiral', 'Wave']
            spiral_total = sum([v for k, v in self.analysis_results['spiral'].items() if isinstance(v, int)])
            wave_total = sum([v for k, v in self.analysis_results['wave'].items() if isinstance(v, int)])
            totals = [spiral_total, wave_total]
            
            bars = axes[1, 1].bar(datasets, totals, color=['#66b3ff', '#ff9999'])
            axes[1, 1].set_title('Total Images by Dataset Type')
            axes[1, 1].set_ylabel('Number of Images')
            
            # Add value labels on bars
            for bar, value in zip(bars, totals):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n📋 COMPREHENSIVE DATASET REPORT")
        print("=" * 60)
        
        report = []
        report.append("# Parkinson's Disease Dataset Analysis Report\n")
        
        # Summary statistics
        if 'spiral' in self.analysis_results:
            spiral_stats = self.analysis_results['spiral']
            total_spiral = sum([v for v in spiral_stats.values() if isinstance(v, int)])
            report.append(f"## Spiral Dataset Summary")
            report.append(f"- Total Images: {total_spiral}")
            report.append(f"- Training Images: {spiral_stats.get('train_healthy', 0) + spiral_stats.get('train_parkinson', 0)}")
            report.append(f"- Testing Images: {spiral_stats.get('test_healthy', 0) + spiral_stats.get('test_parkinson', 0)}")
            report.append(f"- Healthy Images: {spiral_stats.get('train_healthy', 0) + spiral_stats.get('test_healthy', 0)}")
            report.append(f"- Parkinson Images: {spiral_stats.get('train_parkinson', 0) + spiral_stats.get('test_parkinson', 0)}\n")
        
        if 'wave' in self.analysis_results:
            wave_stats = self.analysis_results['wave']
            total_wave = sum([v for v in wave_stats.values() if isinstance(v, int)])
            report.append(f"## Wave Dataset Summary")
            report.append(f"- Total Images: {total_wave}")
            report.append(f"- Training Images: {wave_stats.get('train_healthy', 0) + wave_stats.get('train_parkinson', 0)}")
            report.append(f"- Testing Images: {wave_stats.get('test_healthy', 0) + wave_stats.get('test_parkinson', 0)}")
            report.append(f"- Healthy Images: {wave_stats.get('train_healthy', 0) + wave_stats.get('test_healthy', 0)}")
            report.append(f"- Parkinson Images: {wave_stats.get('train_parkinson', 0) + wave_stats.get('test_parkinson', 0)}\n")
        
        # Recommendations
        report.append("## Recommendations for Vision Transformer Implementation")
        report.append("1. **Data Augmentation**: Consider applying augmentation techniques to increase dataset size")
        report.append("2. **Image Preprocessing**: Standardize image sizes and normalize pixel values")
        report.append("3. **Cross-validation**: Use k-fold cross-validation for robust evaluation")
        report.append("4. **Transfer Learning**: Use pre-trained Vision Transformer models")
        report.append("5. **Class Balancing**: Apply techniques to handle class imbalance if present")
        
        report_text = "\n".join(report)
        
        # Save report
        with open("dataset_analysis_report.md", "w") as f:
            f.write(report_text)
        
        print("✓ Report saved to: dataset_analysis_report.md")
        return report_text
    
    def run_complete_analysis(self):
        """Run complete dataset analysis"""
        print("🚀 Starting Complete Parkinson's Dataset Analysis\n")
        
        # Run all analysis components
        self.analyze_dataset_structure()
        self.analyze_spiral_dataset()
        self.analyze_wave_dataset()
        self.analyze_image_properties()
        self.create_visualization()
        self.generate_report()
        
        print("\n✅ Complete analysis finished!")
        return self.analysis_results

def main():
    """Main function to run the analysis"""
    analyzer = ParkinsonDatasetAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\n🎯 Analysis Results Summary:")
    for dataset_type, stats in results.items():
        if isinstance(stats, dict):
            print(f"\n{dataset_type.upper()}:")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
