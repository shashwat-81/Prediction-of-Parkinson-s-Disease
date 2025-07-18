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

# Import our prediction system
from predict_disease import ParkinsonsPredictor

app = Flask(__name__)
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

def initialize_predictor():
    """Initialize the predictor globally"""
    global predictor
    try:
        if predictor is None:
            predictor = ParkinsonsPredictor()
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
        return render_template('upload.html')
    
    # Check if predictor is initialized
    if not initialize_predictor():
        flash('Prediction system not available. Please check model files.', 'error')
        return redirect(url_for('upload_file'))
    
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('upload_file'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('upload_file'))
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = predictor.predict_single_file(filepath)
            formatted_result = format_prediction_for_web(result)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if formatted_result["success"]:
                return render_template('result.html', 
                                     result=formatted_result, 
                                     filename=file.filename)
            else:
                flash(f'Prediction failed: {formatted_result["error"]}', 'error')
                return redirect(url_for('upload_file'))
                
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('upload_file'))
    else:
        flash('Invalid file type. Please upload WAV, MP3, FLAC, or M4A files.', 'error')
        return redirect(url_for('upload_file'))

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
            
            # Make prediction
            result = predictor.predict_single_file(tmp_file.name)
            formatted_result = format_prediction_for_web(result)
            
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
