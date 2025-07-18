# How to Predict Parkinson's Disease Using Your Voice Data

## Quick Start Guide

### 1. Prepare Your Audio Files
- **Format**: WAV files (preferred), MP3, or FLAC
- **Content**: Record yourself saying "Ah" sound for 3-5 seconds
- **Quality**: Clear recording, minimal background noise
- **Sample Rate**: Any (will be automatically converted to 16kHz)

### 2. Single File Prediction

```python
from predict_disease import ParkinsonsPredictor

# Initialize the predictor
predictor = ParkinsonsPredictor()

# Predict for a single file
result = predictor.predict_single_file("path/to/your/audio.wav")

# Print detailed report
predictor.print_detailed_report(result)
```

### 3. Multiple Files Prediction

```python
# Predict for all audio files in a folder
results = predictor.predict_multiple_files(
    "path/to/audio/folder", 
    output_csv="my_results.csv"  # Optional: save to CSV
)

# Results will be saved to CSV automatically
```

### 4. Command Line Usage

Simply run the script:
```bash
python predict_disease.py
```

Then follow the interactive menu:
1. Choose option 1 for single file
2. Choose option 2 for multiple files
3. Enter file/folder paths when prompted

## Understanding the Results

### Risk Levels
- **Low Risk** (< 30%): Likely healthy
- **Moderate Risk** (30-60%): Consider medical consultation
- **High Risk** (60-80%): Recommend neurological evaluation
- **Very High Risk** (> 80%): Urgent medical attention

### Key Metrics
- **Parkinson's Probability**: 0.0 to 1.0 (higher = more likely)
- **Model Agreement**: How much the different AI models agree (0.0 to 1.0)
- **Prediction Uncertainty**: How confident the prediction is
- **Ensemble vs Single**: Ensemble uses multiple models for better accuracy

## Examples

### Example 1: Low Risk Result
```
🎯 ENSEMBLE PREDICTION
   Predicted Class: HC_AH (Healthy)
   Parkinson's Probability: 0.089
   Healthy Probability: 0.911
   Model Agreement: 0.800
   Risk Level: Low Risk

💡 RECOMMENDATION
   Low concern. Continue regular monitoring.
```

### Example 2: High Risk Result
```
🎯 ENSEMBLE PREDICTION
   Predicted Class: PD_AH (Parkinson's)
   Parkinson's Probability: 0.734
   Healthy Probability: 0.266
   Model Agreement: 0.600
   Risk Level: High Risk

💡 RECOMMENDATION
   High concern. Recommend neurological evaluation.
```

## Important Notes

### ⚠️ Medical Disclaimer
- This is an AI research tool, NOT a medical diagnostic device
- Results should NOT replace professional medical evaluation
- Always consult healthcare professionals for medical decisions
- Use results as supplementary information only

### Best Practices
1. **Multiple Recordings**: Take several recordings for better accuracy
2. **Consistent Conditions**: Record in similar environments
3. **Clear Audio**: Minimize background noise
4. **Regular Monitoring**: Track changes over time
5. **Professional Consultation**: Discuss results with doctors

### Tips for Better Results
- Record in a quiet room
- Use the same microphone/device
- Say "Ah" clearly and steadily
- Record multiple times and compare results
- Keep track of recording dates for monitoring trends

## File Organization

Organize your audio files like this:
```
my_voice_data/
├── 2025-01-15_morning.wav
├── 2025-01-15_evening.wav
├── 2025-02-01_morning.wav
└── results/
    └── predictions_2025-02-01.csv
```

## Troubleshooting

### Common Issues:
1. **"No models found"**: Make sure you're in the correct directory with the .pkl files
2. **"Audio loading error"**: Check file format and path
3. **"Feature extraction failed"**: Audio file might be corrupted
4. **Low model agreement**: Take multiple recordings for verification

### System Requirements:
- Python 3.8+
- At least 4GB RAM
- Internet connection (for first-time model download)
- Microphone for recording audio

## Next Steps

1. **Record Your Voice**: Create audio files saying "Ah"
2. **Run Predictions**: Use the script to analyze your recordings
3. **Track Over Time**: Keep regular recordings to monitor changes
4. **Consult Professionals**: Discuss any concerning results with doctors
5. **Improve Data**: Collect more recordings for better accuracy
