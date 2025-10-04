from predict_parkinsons import MedicalParkinsonPredictor

p = MedicalParkinsonPredictor()
try:
    p.load_model('medical_vit_parkinson_spiral.pth')
    print('Model loaded OK')
except Exception as e:
    print('Load failed:', e)
