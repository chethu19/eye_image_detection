import os
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)

# Model config (same as train script)
IMG_SIZE = (32, 32)
MODEL_PATH = 'eye_state_model.pkl'

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
    return None

@app.route('/')
def index():
    results = ""
    if os.path.exists('training_results.txt'):
        with open('training_results.txt', 'r') as f:
            results = f.read()
    else:
        results = "Training in progress or not started..."
    return render_template('index.html', training_results=results)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model not trained yet. Please wait.'})
            
        img = Image.open(file.stream).convert('L') # Grayscale
        # Resize to match training data
        img = img.resize(IMG_SIZE)
        img_array = np.array(img).flatten() / 255.0
        
        # Get probability
        probs = model.predict_proba([img_array])[0]
        prediction = model.predict([img_array])[0]
        
        state = "Open" if prediction == 1 else "Closed"
        confidence = float(probs[prediction]) # Probability of the predicted class
        
        # Convert original image to base64 for display
        file.stream.seek(0)
        img_orig = Image.open(file.stream)
        buffered = io.BytesIO()
        img_orig.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'prediction': state,
            'confidence': f"{confidence*100:.2f}%",
            'image_data': img_str
        })
    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"})

if __name__ == '__main__':
    # Hugging Face Spaces expects port 7860
    app.run(host='0.0.0.0', port=7860)
