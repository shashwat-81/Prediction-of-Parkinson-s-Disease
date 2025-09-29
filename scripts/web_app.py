"""
Flask Web Application for Parkinson's Disease Voice & Drawing Prediction
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
from PIL import Image

# Import our prediction systems
from predict_disease import ParkinsonsPredictor
from predict_parkinsons import MedicalParkinsonPredictor

app = Flask(__name__, template_folder='templates')
app.secret_key = 'parkinson_prediction_app_2025'
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}
IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | IMAGE_EXTENSIONS
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global predictor instances
voice_predictor = None
drawing_predictor = None

def allowed_file(filename, file_type='any'):
    """Check if file extension is allowed"""
    if not ('.' in filename):
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if file_type == 'audio':
        return ext in AUDIO_EXTENSIONS
    elif file_type == 'image':
        return ext in IMAGE_EXTENSIONS
    else:
        return ext in ALLOWED_EXTENSIONS

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

def initialize_predictors():
    """Initialize both voice and drawing predictors"""
    global voice_predictor, drawing_predictor
    
    try:
        if voice_predictor is None:
            # Use absolute path to scripts directory for model files
            scripts_dir = os.path.dirname(os.path.abspath(__file__))
            voice_predictor = ParkinsonsPredictor(models_dir=scripts_dir)
            print("✅ Voice predictor initialized")
    except Exception as e:
        print(f"❌ Voice predictor initialization failed: {e}")
    
    try:
        if drawing_predictor is None:
            drawing_predictor = MedicalParkinsonPredictor()
            model_path = "medical_vit_parkinson_spiral.pth"
            if os.path.exists(model_path):
                drawing_predictor.load_model(model_path)
                print("✅ Drawing predictor initialized")
            else:
                print("⚠️ Drawing model not found, drawing prediction disabled")
                drawing_predictor = None
    except Exception as e:
        print(f"❌ Drawing predictor initialization failed: {e}")
        drawing_predictor = None
    
    return voice_predictor is not None or drawing_predictor is not None

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

# Routes
@app.route('/')
def index():
    """Main page with both voice and drawing options"""
    return render_template('index.html')

@app.route('/upload')
def upload_redirect():
    """Redirect old upload URL to voice upload"""
    return redirect(url_for('upload_voice'))

@app.route('/upload/voice')
def upload_voice():
    """Voice upload page"""
    return render_template('upload_voice.html')

@app.route('/upload/voice', methods=['POST'])
def process_voice_upload():
    """Process voice file upload"""
    if not voice_predictor:
        flash('Voice analysis is not available. Please check if the model files exist.', 'error')
        return render_template('upload_voice.html')
    
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return render_template('upload_voice.html')
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return render_template('upload_voice.html')
    
    if file and allowed_file(file.filename, 'audio'):
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
                flash(f'Upload rejected: {reason}. Please upload a simple voice recording.', 'error')
                return render_template('upload_voice.html')
            
            # Make prediction
            result = voice_predictor.predict_single_file(filepath)
            formatted_result = format_prediction_for_web(result)
            
            # Add audio analysis info
            formatted_result["audio_analysis"] = {
                "duration": audio_info.get("duration", 0),
                "complexity_score": audio_info.get("complexity_score", 0),
                "analysis_note": reason
            }
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if formatted_result["success"]:
                return render_template('voice_result.html', result=formatted_result, filename=file.filename)
            else:
                flash(f'Prediction failed: {formatted_result["error"]}', 'error')
                return render_template('upload_voice.html')
                
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return render_template('upload_voice.html')
    else:
        flash('Invalid file type. Please upload WAV, MP3, FLAC, or M4A files.', 'error')
        return render_template('upload_voice.html')

@app.route('/upload/drawing')
def upload_drawing():
    """Drawing upload page"""
    return render_template('upload_drawing.html')

@app.route('/upload/drawing', methods=['POST'])
def process_drawing_upload():
    """Process drawing file upload"""
    if not drawing_predictor:
        flash('Drawing analysis is not available. Please check if the model file exists.', 'error')
        return render_template('upload_drawing.html')
    
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return render_template('upload_drawing.html')
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return render_template('upload_drawing.html')
    
    if file and allowed_file(file.filename, 'image'):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Validate it's actually an image
            try:
                img = Image.open(filepath)
                img.verify()
            except Exception:
                os.remove(filepath)
                flash('Invalid image file. Please upload a valid PNG, JPG, or JPEG image.', 'error')
                return render_template('upload_drawing.html')
            
            # Make prediction
            result = drawing_predictor.predict_single_image(filepath, return_details=True)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if result:
                return render_template('drawing_result.html', result=result, filename=file.filename)
            else:
                flash('Failed to analyze the drawing. Please try with a different image.', 'error')
                return render_template('upload_drawing.html')
                
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return render_template('upload_drawing.html')
    else:
        flash('Invalid file type. Please upload PNG, JPG, or JPEG images.', 'error')
        return render_template('upload_drawing.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if not voice_predictor:
        return jsonify({"success": False, "error": "Voice prediction system not available"})
    
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename, 'audio'):
        return jsonify({"success": False, "error": "Invalid file"})
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            file.save(tmp_file.name)
            
            is_music, reason, audio_info = is_music_or_complex_audio(tmp_file.name)
            
            if is_music:
                os.unlink(tmp_file.name)
                return jsonify({
                    "success": False, 
                    "error": f"Music/complex audio rejected: {reason}",
                    "audio_analysis": audio_info
                })
            
            result = voice_predictor.predict_single_file(tmp_file.name)
            formatted_result = format_prediction_for_web(result)
            
            formatted_result["audio_analysis"] = {
                "duration": audio_info.get("duration", 0),
                "complexity_score": audio_info.get("complexity_score", 0),
                "analysis_note": reason
            }
            
            os.unlink(tmp_file.name)
            return jsonify(formatted_result)
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        predictor_status = initialize_predictors()
        return jsonify({
            "status": "healthy" if predictor_status else "degraded",
            "voice_predictor_available": voice_predictor is not None,
            "drawing_predictor_available": drawing_predictor is not None,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    print("🎤🖊️ Starting Parkinson's Disease Prediction Web App")
    print("=" * 60)
    
    if initialize_predictors():
        print("✅ Prediction systems initialized successfully")
    else:
        print("❌ Warning: Some prediction systems failed to initialize")
    
    print("🌐 Web app will be available at: http://localhost:5000")
    print("📁 Upload folder:", os.path.abspath(UPLOAD_FOLDER))
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)