from flask import Flask, render_template, request, redirect, url_for
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from joblib import load
import io
import os

app = Flask(__name__)

# Load model configurations
with open('model_config.json') as f:
    model_configs = json.load(f)

# Dictionary to store loaded models
loaded_models = {}


def load_model(disease_type):
    """Load model if not already loaded"""
    if disease_type not in loaded_models:
        config = model_configs[disease_type]
        model_path = config['model_path']

        if config['model_type'] == 'tensorflow':
            model = tf.keras.models.load_model(model_path)
        else:  # sklearn
            model = load(model_path)

        loaded_models[disease_type] = model

    return loaded_models[disease_type]


def preprocess_image(image_file, config):
    """Preprocess image according to model requirements"""
    # Read image
    image = Image.open(image_file)

    # Resize
    target_size = config['preprocessing']['target_size']
    image = image.resize(target_size)

    # Convert to array and add batch dimension
    img_array = np.array(image)
    #img_array = np.expand_dims(img_array, axis=0)

    # Convert grayscale to RGB by repeating the channels
    #img_resized = np.repeat(image, 3, axis=-1)
    if img_array.ndim == 2:  # If the image is grayscale (2D)
        img_array = np.repeat(img_array[..., np.newaxis], 3, axis=-1)  # Convert to RGB by repeating channels

    # Normalize if required
    if config['preprocessing'].get('normalize', False):
        img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def preprocess_tabular_data(form_data, config):
    """Preprocess tabular data according to model requirements"""
    # Get input order from config
    input_order = config['preprocessing']['input_order']

    # Create feature vector in correct order
    features = []
    for field in input_order:
        value = form_data.get(field)
        if value is not None:
            features.append(float(value))

    return np.array([features])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/<disease_type>', methods=['GET', 'POST'])
def predict(disease_type):
    # Check if disease type exists in config
    if disease_type not in model_configs:
        return redirect(url_for('home'))

    config = model_configs[disease_type]
    prediction = None
    error = None
    confidence = None

    try:
        if request.method == 'POST':
            # Load model
            model = load_model(disease_type)

            if config['data_type'] == 'image':
                # Handle image upload
                if 'image' not in request.files:
                    return redirect(request.url)

                file = request.files['image']
                if file.filename == '':
                    return redirect(request.url)

                if file:
                    # Preprocess image
                    img_array = preprocess_image(file, config)

                    # Make prediction
                    pred_array = model.predict(img_array)

                    # Process prediction based on output format
                    if config['output_format']['type'] == 'binary':
                        threshold = config['output_format'].get('threshold', 0.5)

                        # Handle different shapes of prediction arrays
                        if pred_array.ndim > 1:
                            pred_value = pred_array[0].max()  # Get the highest probability if multiple classes
                            prediction_class = pred_array[0].argmax()
                            confidence = float(pred_value)
                            prediction = 1 if pred_value >= threshold else 0
                        else:
                            pred_value = float(pred_array[0])
                            confidence = pred_value
                            prediction = 1 if pred_value >= threshold else 0

                    elif config['output_format']['type'] == 'multiclass':
                        prediction = int(pred_array[0].argmax())
                        confidence = float(pred_array[0].max())

            else:  # tabular data
                # Get form data
                form_data = request.form.to_dict()

                # Preprocess tabular data
                features = preprocess_tabular_data(form_data, config)

                # Make prediction
                pred_array = model.predict(features)

                # Process prediction based on output format
                if config['output_format']['type'] == 'binary':
                    pred_value = float(pred_array[0])
                    confidence = pred_value
                    prediction = 1 if pred_value >= 0.5 else 0
                elif config['output_format']['type'] == 'multiclass':
                    prediction = int(pred_array.argmax())
                    confidence = float(pred_array.max())

    except Exception as e:
        error = str(e)
        print(f"Error in prediction: {error}")
        # Log additional debug information
        import traceback
        print(traceback.format_exc())

    return render_template(
        'predict.html',
        disease_type=disease_type,
        prediction=prediction,
        confidence=confidence,
        error=error,
        config=config
    )


if __name__ == '__main__':
    app.run(debug=True)