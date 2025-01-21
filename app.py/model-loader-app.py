# app.py
from flask import Flask, render_template, request, jsonify
import os

from keras.src.utils.module_utils import tensorflow
from werkzeug.utils import secure_filename
import tensorflow as tf
import joblib
import pickle
import numpy as np
import json
import cv2

from data_preprocessing import DataPreprocessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model configurations
with open('model_config.json', 'r') as f:
    MODEL_CONFIG = json.load(f)


def _preprocess_image(image_path, preprocessing_config):
    """Preprocess image based on model requirements"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (preprocessing_config['target_size']))

    if preprocessing_config.get('normalize', True):
        img = img / 255.0

    return img


class ModelManager:
    def __init__(self):
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all models at startup"""
        for disease, config in MODEL_CONFIG.items():
            if config['model_type'] == 'tensorflow':
                self.models[disease] = tensorflow.keras.models.load_model(config['model_path'])
            elif config['model_type'] == 'sklearn':
                self.models[disease] = joblib.load(config['model_path'])


    
    def predict(self, disease, data):
        """Make prediction using appropriate model"""
        model = self.models[disease]
        config = MODEL_CONFIG[disease]
        
        if config['data_type'] == 'image':
            # Preprocess image according to model requirements
            processed_data = _preprocess_image(data, config['preprocessing'])
            prediction = model.predict(np.expand_dims(processed_data, axis=0))
            
        else:  # tabular data
            # Preprocess tabular data according to model requirements
            processed_data = self._preprocess_tabular(data, config['preprocessing'])
            prediction = model.predict([processed_data])
        
        return self._format_prediction(prediction, config['output_format'])

    def _preprocess_tabular(self, data, preprocessing_config):
        """Preprocess tabular data based on model requirements"""
        # Convert input data to correct types
        processed = []
        for field in preprocessing_config['input_order']:
            value = data.get(field, 0)
            processed.append(float(value))
        
        return processed
    
    def _format_prediction(self, prediction, output_format):
        """Format prediction based on configuration"""
        if output_format.get('type') == 'binary':
            return {
                'prediction': bool(prediction[0] > output_format.get('threshold', 0.5)),
                'confidence': float(prediction[0])
            }
        elif output_format.get('type') == 'categorical':
            pred_class = int(np.argmax(prediction[0]))
            return {
                'prediction': output_format['classes'][pred_class],
                'confidence': float(prediction[0][pred_class])
            }
        else:
            return {'prediction': float(prediction[0])}

# Initialize model manager
model_manager = ModelManager()

@app.route('/')
def home():
    """Home page showing available disease detection options"""
    return render_template('index.html', diseases=MODEL_CONFIG)

@app.route('/predict/<disease_name>', methods=['GET', 'POST'])
def predict(disease_name):
    if disease_name not in MODEL_CONFIG:
        return jsonify({'error': 'Disease not found'})
    
    disease_info = MODEL_CONFIG[disease_name]
    
    if request.method == 'GET':
        return render_template('predict.html', 
                             disease=disease_name,
                             disease_info=disease_info)
    
    try:
        if disease_info['data_type'] == 'image':
            if 'image' not in request.files:
                return jsonify({'error': 'No image uploaded'})
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No selected file'})
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction
            result = model_manager.predict(disease_name, filepath)
            
            # Clean up
            os.remove(filepath)
            
        else:  # tabular data
            # Get form data
            input_data = request.form.to_dict()
            result = model_manager.predict(disease_name, input_data)
        
        return jsonify(result)
            
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
