"""
Flask Web Application for Parkinson's Disease Voice Prediction
"""

import os
import tempfile
import shutil
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import librosa
import numpy as np

# Import our prediction system
from predict_disease import ParkinsonsPredictor

app = Flask(__name__, template_folder='templates')
app.secret_key = 'parkinson_prediction_app_2025'
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global predictor instance
predictor = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_music_or_complex_audio(audio_path, threshold_complexity=0.6, min_duration=1.0):
    """
    Detect if audio is music/complex audio vs simple voice recording
    
    Args:
        audio_path (str): Path to audio file
        threshold_complexity (float): Complexity threshold (higher = more complex)
        min_duration (float): Minimum duration in seconds
    
    Returns:
        tuple: (is_music, reason, audio_info)
    """
    try:
        # Load audio with error handling
        try:
            y, sr = librosa.load(audio_path, sr=None, duration=30)  # Load max 30 seconds
        except Exception as load_error:
            return True, f"Could not load audio file: {str(load_error)}", {"error": "audio_load_failed"}
        
        duration = len(y) / sr
        
        # Check minimum duration
        if duration < min_duration:
            return True, f"Audio too short ({duration:.1f}s). Need at least {min_duration}s", {"duration": float(duration)}
        
        # Feature extraction for music detection
        try:
            # 1. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            # 2. Rhythm features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # 3. Harmonic features
            harmonic, percussive = librosa.effects.hpss(y)
            harmonic_ratio = np.mean(np.abs(harmonic)) / (np.mean(np.abs(y)) + 1e-8)
            
            # 4. Chroma features (for musical content)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_std = np.std(chroma)
            
            # 5. Zero crossing rate (voice vs music)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            
            # 6. MFCCs variation
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_std = np.mean(np.std(mfccs, axis=1))
            
            # 7. Energy variation
            energy = librosa.feature.rms(y=y)[0]
            energy_std = np.std(energy)
            
        except Exception as feature_error:
            return True, f"Could not extract audio features: {str(feature_error)}", {"error": "feature_extraction_failed"}
        
        # Music detection logic
        complexity_score = 0
        reasons = []
        
        # High spectral complexity suggests music (raised threshold)
        if np.std(spectral_centroids) > 1500:  # Raised from 1000 to 1500
            complexity_score += 0.25  # Reduced impact from 0.3 to 0.25
            reasons.append("Very high spectral variation")
        
        # Strong rhythmic content suggests music  
        if tempo > 60 and len(beats) > 10:
            complexity_score += 0.25
            reasons.append(f"Musical tempo detected ({float(tempo):.0f} BPM)")
        
        # High harmonic content suggests music (more permissive for voice)
        if harmonic_ratio > 0.8:  # Raised from 0.6 to 0.8 - voice can be harmonic
            complexity_score += 0.15  # Reduced impact from 0.2 to 0.15
            reasons.append("Very strong harmonic content")
        
        # Musical chroma patterns (more permissive for voice harmonics)
        if chroma_std > 0.25:  # Raised from 0.15 to 0.25 - natural voice harmonics
            complexity_score += 0.15  # Reduced impact from 0.2 to 0.15
            reasons.append("Complex musical chord patterns")
        
        # Complex MFCC patterns suggest music
        if mfcc_std > 15:
            complexity_score += 0.15
            reasons.append("Complex spectral patterns")
        
        # Very low or very high ZCR suggests non-voice
        if zcr_mean < 0.01 or zcr_mean > 0.3:
            complexity_score += 0.1
            reasons.append("Unusual voice characteristics")
        
        # Audio info for debugging
        audio_info = {
            "duration": float(duration),
            "tempo": float(tempo),
            "complexity_score": float(complexity_score),
            "spectral_centroid_std": float(np.std(spectral_centroids)),
            "harmonic_ratio": float(harmonic_ratio),
            "chroma_std": float(chroma_std),
            "mfcc_std": float(mfcc_std),
            "zcr_mean": float(zcr_mean)
        }
        
        is_music = complexity_score > threshold_complexity
        
        if is_music:
            reason = f"Detected as music/complex audio (score: {complexity_score:.2f}). " + "; ".join(reasons)
        else:
            reason = f"Appears to be voice recording (score: {complexity_score:.2f})"
            
        return is_music, reason, audio_info
        
    except Exception as e:
        return True, f"Error analyzing audio: {str(e)}", {"error": str(e)}

def initialize_predictor():
    """Initialize the predictor globally"""
    global predictor
    try:
        if predictor is None:
            # Use absolute path to scripts directory for model files
            scripts_dir = os.path.dirname(os.path.abspath(__file__))
            predictor = ParkinsonsPredictor(models_dir=scripts_dir)
        return True
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return False

def format_prediction_for_web(result):
    """Format prediction result for web display"""
    if "error" in result:
        return {"success": False, "error": result["error"]}
    
    # Determine if ensemble or single model result
    if "ensemble_prediction" in result:
        formatted = {
            "success": True,
            "prediction_type": "ensemble",
            "predicted_class": result["ensemble_prediction"],
            "parkinson_probability": round(result["ensemble_parkinson_probability"] * 100, 1),
            "healthy_probability": round(result["ensemble_healthy_probability"] * 100, 1),
            "risk_level": result["risk_level"],
            "model_agreement": round(result["model_agreement"] * 100, 1),
            "uncertainty": round(result["prediction_uncertainty"] * 100, 1),
            "recommendation": result["recommendation"],
            "individual_models": []
        }
        
        # Add individual model results
        for model_name, pred in result["individual_predictions"].items():
            formatted["individual_models"].append({
                "name": model_name,
                "prediction": pred["predicted_class"],
                "parkinson_prob": round(pred["parkinson_probability"] * 100, 1),
                "confidence": round(pred["confidence"] * 100, 1)
            })
    else:
        formatted = {
            "success": True,
            "prediction_type": "single",
            "predicted_class": result["prediction"],
            "parkinson_probability": round(result["parkinson_probability"] * 100, 1),
            "healthy_probability": round(result["healthy_probability"] * 100, 1),
            "risk_level": result["risk_level"],
            "confidence": round(result["confidence"] * 100, 1),
            "recommendation": result["recommendation"],
            "model_used": result["model_used"]
        }
    
    return formatted

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and prediction"""
    if request.method == 'GET':
        # Return working HTML directly - bypass template issues
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parkinson's Voice Analysis - Upload</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            margin: 20px auto;
            max-width: 800px;
            padding: 40px;
        }
        .upload-zone {
            border: 3px dashed #007bff;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9fa;
            margin: 20px 0;
            transition: all 0.3s ease;
        }
        .upload-zone:hover {
            border-color: #0056b3;
            background: #e3f2fd;
        }
        .btn-primary {
            background: #007bff;
            border: none;
            border-radius: 8px;
            padding: 12px 30px;
            font-size: 16px;
        }
        .alert-success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-container">
            <div class="text-center mb-4">
                <h1 class="fw-bold text-primary">
                    <i class="fas fa-microphone me-3"></i>
                    Parkinson's Voice Analysis
                </h1>
                <p class="lead text-muted">Upload your voice recording for AI-powered analysis</p>
            </div>

            <div class="alert-success">
                <h6><i class="fas fa-info-circle me-2"></i>Voice Recording Guidelines</h6>
                <ul class="mb-0">
                    <li><strong>✅ Upload:</strong> Simple voice recordings (saying "ahhhh" for 3+ seconds)</li>
                    <li><strong>✅ Acceptable:</strong> Speech, sustained vowels, voice exercises</li>
                    <li><strong>❌ Rejected:</strong> Music, songs, complex audio with instruments</li>
                    <li><strong>📏 Duration:</strong> At least 1 second, recommended 3+ seconds</li>
                    <li><strong>🎤 Quality:</strong> Clear recording without background music</li>
                </ul>
            </div>

            <div class="upload-zone">
                <form method="post" enctype="multipart/form-data" id="uploadForm">
                    <i class="fas fa-cloud-upload-alt fa-3x text-primary mb-3"></i>
                    <h4>Upload Your Voice Recording</h4>
                    <p class="text-muted mb-4">Drag and drop a file here or click to browse</p>
                    
                    <input type="file" name="file" id="fileInput" accept=".wav,.mp3,.flac,.m4a" required 
                           class="form-control mb-3" style="max-width: 400px; margin: 0 auto;">
                    
                    <button type="submit" class="btn btn-primary btn-lg">
                        <i class="fas fa-brain me-2"></i>
                        Analyze Voice
                    </button>
                </form>
            </div>

            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title">
                                <i class="fas fa-check-circle text-success me-2"></i>
                                What We Accept
                            </h6>
                            <ul class="list-unstyled">
                                <li>• Voice recordings</li>
                                <li>• Sustained vowels ("ahhhh")</li>
                                <li>• Speech samples</li>
                                <li>• Clear audio files</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title">
                                <i class="fas fa-times-circle text-danger me-2"></i>
                                What We Reject
                            </h6>
                            <ul class="list-unstyled">
                                <li>• Music files</li>
                                <li>• Songs with instruments</li>
                                <li>• Complex audio</li>
                                <li>• Background music</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <div class="text-center mt-4">
                <small class="text-muted">
                    <i class="fas fa-shield-alt me-1"></i>
                    Your voice data is processed securely and not stored permanently
                </small>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''
    
    # POST request - handle file upload
    # Check if predictor is initialized
    if not initialize_predictor():
        error_msg = 'Prediction system not available. Please check model files.'
        return get_upload_page_with_message(error_msg, 'error')
    
    # Check if file was uploaded
    if 'file' not in request.files:
        error_msg = 'No file selected'
        return get_upload_page_with_message(error_msg, 'error')
    
    file = request.files['file']
    
    if file.filename == '':
        error_msg = 'No file selected'
        return get_upload_page_with_message(error_msg, 'error')
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Check if audio is music/complex audio
            is_music, reason, audio_info = is_music_or_complex_audio(filepath)
            
            if is_music:
                # Clean up uploaded file
                os.remove(filepath)
                error_msg = f'❌ Upload rejected: {reason}. Please upload a simple voice recording (like saying "ahhhh" for 3+ seconds).'
                return get_upload_page_with_message(error_msg, 'error')
            
            # Make prediction
            result = predictor.predict_single_file(filepath)
            formatted_result = format_prediction_for_web(result)
            
            # Add audio analysis info to result
            formatted_result["audio_analysis"] = {
                "duration": audio_info.get("duration", 0),
                "complexity_score": audio_info.get("complexity_score", 0),
                "analysis_note": reason
            }
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if formatted_result["success"]:
                # Show results on the same page
                return get_results_page(formatted_result, file.filename)
            else:
                error_msg = f'Prediction failed: {formatted_result["error"]}'
                return get_upload_page_with_message(error_msg, 'error')
                
        except Exception as e:
            error_msg = f'Error processing file: {str(e)}'
            return get_upload_page_with_message(error_msg, 'error')
    else:
        error_msg = 'Invalid file type. Please upload WAV, MP3, FLAC, or M4A files.'
        return get_upload_page_with_message(error_msg, 'error')

def get_results_page(result, filename):
    """Return results page with prediction results"""
    # Determine risk level color
    risk_color = "danger" if result["predicted_class"] == "Parkinson's" else "success"
    risk_icon = "fas fa-exclamation-triangle" if result["predicted_class"] == "Parkinson's" else "fas fa-check-circle"
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Results - Parkinson's Voice AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .main-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            margin: 20px auto;
            max-width: 900px;
            padding: 40px;
        }}
        .result-card {{
            border: none;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }}
        .progress-custom {{
            height: 25px;
            border-radius: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="main-container">
            <div class="text-center mb-4">
                <h1 class="fw-bold text-primary">
                    <i class="fas fa-chart-line me-3"></i>
                    Voice Analysis Results
                </h1>
                <p class="lead text-muted">Analysis for: <strong>{filename}</strong></p>
            </div>

            <!-- Main Result -->
            <div class="card result-card">
                <div class="card-body text-center p-5">
                    <div class="alert alert-{risk_color} mb-4">
                        <i class="{risk_icon} fa-2x mb-3"></i>
                        <h3 class="mb-2">{result["predicted_class"]}</h3>
                        <p class="mb-0">Confidence: {result.get("parkinson_probability", 0)}%</p>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Parkinson's Probability</h6>
                            <div class="progress progress-custom mb-3">
                                <div class="progress-bar bg-danger" style="width: {result.get('parkinson_probability', 0)}%">
                                    {result.get('parkinson_probability', 0)}%
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <h6>Healthy Probability</h6>
                            <div class="progress progress-custom mb-3">
                                <div class="progress-bar bg-success" style="width: {result.get('healthy_probability', 0)}%">
                                    {result.get('healthy_probability', 0)}%
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Model Details -->
            {get_model_details_html(result)}

            <!-- Audio Analysis -->
            <div class="card result-card">
                <div class="card-body">
                    <h5 class="card-title">
                        <i class="fas fa-waveform-lines me-2"></i>
                        Audio Analysis
                    </h5>
                    <div class="row">
                        <div class="col-md-4">
                            <small class="text-muted">Duration</small>
                            <p class="mb-0">{result["audio_analysis"]["duration"]:.1f} seconds</p>
                        </div>
                        <div class="col-md-4">
                            <small class="text-muted">Complexity Score</small>
                            <p class="mb-0">{result["audio_analysis"]["complexity_score"]:.2f}</p>
                        </div>
                        <div class="col-md-4">
                            <small class="text-muted">Analysis Note</small>
                            <p class="mb-0">{result["audio_analysis"]["analysis_note"]}</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Disclaimer -->
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Medical Disclaimer:</strong> This AI analysis is for informational purposes only and should not replace professional medical diagnosis. Please consult with a healthcare provider for proper medical evaluation.
            </div>

            <!-- Action Buttons -->
            <div class="text-center">
                <a href="/upload" class="btn btn-primary btn-lg me-3">
                    <i class="fas fa-upload me-2"></i>
                    Analyze Another Recording
                </a>
                <a href="/" class="btn btn-outline-primary btn-lg">
                    <i class="fas fa-home me-2"></i>
                    Home
                </a>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''

def get_model_details_html(result):
    """Generate HTML for model details section"""
    if result.get("prediction_type") == "ensemble":
        models_html = ""
        for model in result.get("individual_models", []):
            models_html += f'''
            <div class="col-md-6 mb-3">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">{model["name"]}</h6>
                        <p class="mb-1">Prediction: <strong>{model["prediction"]}</strong></p>
                        <p class="mb-0">Confidence: {model["confidence"]}%</p>
                    </div>
                </div>
            </div>
            '''
        
        return f'''
        <div class="card result-card">
            <div class="card-body">
                <h5 class="card-title">
                    <i class="fas fa-brain me-2"></i>
                    Ensemble Model Analysis
                </h5>
                <div class="row mb-3">
                    <div class="col-md-6">
                        <small class="text-muted">Model Agreement</small>
                        <p class="mb-0">{result.get("model_agreement", 0)}%</p>
                    </div>
                    <div class="col-md-6">
                        <small class="text-muted">Prediction Uncertainty</small>
                        <p class="mb-0">{result.get("uncertainty", 0)}%</p>
                    </div>
                </div>
                <h6>Individual Model Results:</h6>
                <div class="row">
                    {models_html}
                </div>
            </div>
        </div>
        '''
    else:
        return f'''
        <div class="card result-card">
            <div class="card-body">
                <h5 class="card-title">
                    <i class="fas fa-brain me-2"></i>
                    Model Analysis
                </h5>
                <p>Model Used: <strong>{result.get("model_used", "Unknown")}</strong></p>
                <p>Confidence: <strong>{result.get("confidence", 0)}%</strong></p>
            </div>
        </div>
        '''

def get_upload_page_with_message(message, msg_type):
    alert_class = "alert-danger" if msg_type == 'error' else "alert-success"
    alert_icon = "fas fa-exclamation-triangle" if msg_type == 'error' else "fas fa-check-circle"
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parkinson's Voice Analysis - Upload</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .main-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            margin: 20px auto;
            max-width: 800px;
            padding: 40px;
        }}
        .upload-zone {{
            border: 3px dashed #007bff;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9fa;
            margin: 20px 0;
            transition: all 0.3s ease;
        }}
        .upload-zone:hover {{
            border-color: #0056b3;
            background: #e3f2fd;
        }}
        .btn-primary {{
            background: #007bff;
            border: none;
            border-radius: 8px;
            padding: 12px 30px;
            font-size: 16px;
        }}
        .alert-success {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .alert-danger {{
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="main-container">
            <div class="text-center mb-4">
                <h1 class="fw-bold text-primary">
                    <i class="fas fa-microphone me-3"></i>
                    Parkinson's Voice Analysis
                </h1>
                <p class="lead text-muted">Upload your voice recording for AI-powered analysis</p>
            </div>

            <div class="alert {alert_class}">
                <i class="{alert_icon} me-2"></i>
                {message}
            </div>

            <div class="alert-success">
                <h6><i class="fas fa-info-circle me-2"></i>Voice Recording Guidelines</h6>
                <ul class="mb-0">
                    <li><strong>✅ Upload:</strong> Simple voice recordings (saying "ahhhh" for 3+ seconds)</li>
                    <li><strong>✅ Acceptable:</strong> Speech, sustained vowels, voice exercises</li>
                    <li><strong>❌ Rejected:</strong> Music, songs, complex audio with instruments</li>
                    <li><strong>📏 Duration:</strong> At least 1 second, recommended 3+ seconds</li>
                    <li><strong>🎤 Quality:</strong> Clear recording without background music</li>
                </ul>
            </div>

            <div class="upload-zone">
                <form method="post" enctype="multipart/form-data" id="uploadForm">
                    <i class="fas fa-cloud-upload-alt fa-3x text-primary mb-3"></i>
                    <h4>Upload Your Voice Recording</h4>
                    <p class="text-muted mb-4">Drag and drop a file here or click to browse</p>
                    
                    <input type="file" name="file" id="fileInput" accept=".wav,.mp3,.flac,.m4a" required 
                           class="form-control mb-3" style="max-width: 400px; margin: 0 auto;">
                    
                    <button type="submit" class="btn btn-primary btn-lg">
                        <i class="fas fa-brain me-2"></i>
                        Analyze Voice
                    </button>
                </form>
            </div>

            <div class="text-center mt-4">
                <a href="/upload" class="btn btn-outline-primary">
                    <i class="fas fa-refresh me-2"></i>
                    Try Again
                </a>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if not initialize_predictor():
        return jsonify({"success": False, "error": "Prediction system not available"})
    
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file"})
    
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            file.save(tmp_file.name)
            
            # Check if audio is music/complex audio
            is_music, reason, audio_info = is_music_or_complex_audio(tmp_file.name)
            
            if is_music:
                # Clean up
                os.unlink(tmp_file.name)
                return jsonify({
                    "success": False, 
                    "error": f"Music/complex audio rejected: {reason}",
                    "audio_analysis": audio_info
                })
            
            # Make prediction
            result = predictor.predict_single_file(tmp_file.name)
            formatted_result = format_prediction_for_web(result)
            
            # Add audio analysis info
            formatted_result["audio_analysis"] = {
                "duration": audio_info.get("duration", 0),
                "complexity_score": audio_info.get("complexity_score", 0),
                "analysis_note": reason
            }
            
            # Clean up
            os.unlink(tmp_file.name)
            
            return jsonify(formatted_result)
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/batch')
def batch_upload():
    """Page for batch predictions"""
    return render_template('batch.html')

@app.route('/about')
def about():
    """About page with information"""
    return render_template('about.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        predictor_status = initialize_predictor()
        return jsonify({
            "status": "healthy" if predictor_status else "degraded",
            "predictor_available": predictor_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    print("🎤 Starting Parkinson's Disease Prediction Web App")
    print("=" * 50)
    
    # Initialize predictor on startup
    if initialize_predictor():
        print("✅ Prediction system initialized successfully")
    else:
        print("❌ Warning: Prediction system failed to initialize")
        print("   Make sure model files are present in the scripts directory")
    
    print("🌐 Web app will be available at: http://localhost:5000")
    print("📁 Upload folder:", os.path.abspath(UPLOAD_FOLDER))
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
