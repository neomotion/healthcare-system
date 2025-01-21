import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import yaml
from typing import List, Dict, Union, Tuple, Optional
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import json
import mlflow
from tensorflow.keras.models import load_model

class UnifiedDevelopmentPipeline:
    """
    A unified development pipeline for both image and tabular models.
    Handles model testing, evaluation, versioning, and deployment.
    """
    
    def __init__(self, config_path: str):
        """Initialize the development pipeline with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_mlflow()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = "logs/development"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"development_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, log_filename)),
                logging.StreamHandler()
            ]
        )
    
    def _setup_mlflow(self):
        """Configure MLflow for experiment tracking."""
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def run_development(self, model_path: str, test_data_path: str):
        """
        Main entry point for development pipeline.
        
        Args:
            model_path: Path to the trained model
            test_data_path: Path to test data
        """
        try:
            model_type = self._detect_model_type(model_path)
            logging.info(f"Detected model type: {model_type}")
            
            if model_type == 'image':
                self._run_image_development(model_path, test_data_path)
            else:
                self._run_tabular_development(model_path, test_data_path)
                
        except Exception as e:
            logging.error(f"Development pipeline failed: {str(e)}")
            raise

    def _detect_model_type(self, model_path: str) -> str:
        """
        Detect whether the model is for image or tabular data.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            str: 'image' or 'tabular'
        """
        if model_path.endswith('.h5'):
            return 'image'
        elif model_path.endswith('.pkl'):
            return 'tabular'
        else:
            raise ValueError("Unknown model type. Model should be .h5 for images or .pkl for tabular data.")

    def _run_image_development(self, model_path: str, test_data_path: str):
        """
        Run development pipeline for image models.
        
        Args:
            model_path: Path to the trained image model
            test_data_path: Path to test image directory
        """
        logging.info("Starting image model development pipeline")
        
        # Load model
        model = load_model(model_path)
        
        # Load and preprocess test data
        test_data = self._load_test_images(test_data_path)
        
        # Evaluate model
        metrics = self._evaluate_image_model(model, test_data)
        
        # Log metrics
        self._log_metrics(metrics, 'image')
        
        # Version the model
        version = self._version_model(model_path, metrics)
        
        # Run performance tests
        self._test_model_performance(model, test_data, 'image')
        
        # Generate deployment artifacts
        self._prepare_deployment(model, 'image', version)
        
        logging.info("Image model development pipeline completed")

    def _run_tabular_development(self, model_path: str, test_data_path: str):
        """
        Run development pipeline for tabular models.
        
        Args:
            model_path: Path to the trained tabular model
            test_data_path: Path to test data file
        """
        logging.info("Starting tabular model development pipeline")
        
        # Load model
        model = self._load_tabular_model(model_path)
        
        # Load and preprocess test data
        test_data = self._load_test_tabular(test_data_path)
        
        # Evaluate model
        metrics = self._evaluate_tabular_model(model, test_data)
        
        # Log metrics
        self._log_metrics(metrics, 'tabular')
        
        # Version the model
        version = self._version_model(model_path, metrics)
        
        # Run performance tests
        self._test_model_performance(model, test_data, 'tabular')
        
        # Generate deployment artifacts
        self._prepare_deployment(model, 'tabular', version)
        
        logging.info("Tabular model development pipeline completed")

    def _load_test_images(self, test_data_path: str) -> Dict[str, np.ndarray]:
        """Load and preprocess test images."""
        images = []
        labels = []
        label_names = []
        
        for label in os.listdir(test_data_path):
            label_dir = os.path.join(test_data_path, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(label_dir, img_file)
                        img = tf.keras.preprocessing.image.load_img(
                            img_path, 
                            target_size=tuple(self.config['image']['img_size'])
                        )
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        images.append(img_array)
                        labels.append(len(label_names))
                        if label not in label_names:
                            label_names.append(label)
        
        return {
            'images': np.array(images),
            'labels': np.array(labels),
            'label_names': label_names
        }

    def _evaluate_image_model(self, model, test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate image model performance."""
        predictions = model.predict(test_data['images'])
        pred_classes = np.argmax(predictions, axis=1)
        
        metrics = {
            'accuracy': float(np.mean(pred_classes == test_data['labels'])),
            'confusion_matrix': confusion_matrix(test_data['labels'], pred_classes).tolist(),
            'classification_report': classification_report(
                test_data['labels'],
                pred_classes,
                target_names=test_data['label_names'],
                output_dict=True
            )
        }
        
        return metrics

    def _test_model_performance(self, model, test_data: Dict[str, np.ndarray], model_type: str):
        """Test model performance characteristics."""
        # Memory usage test
        memory_usage = self._measure_memory_usage(model)
        
        # Inference speed test
        if model_type == 'image':
            batch_sizes = [1, 4, 8, 16, 32]
            inference_times = self._measure_inference_speed(
                model, 
                test_data['images'], 
                batch_sizes
            )
        else:
            batch_sizes = [1000, 5000, 10000]
            inference_times = self._measure_inference_speed(
                model, 
                test_data['features'], 
                batch_sizes
            )
        
        # Log performance metrics
        with mlflow.start_run(nested=True):
            mlflow.log_metric("memory_usage_mb", memory_usage)
            for batch_size, time in zip(batch_sizes, inference_times):
                mlflow.log_metric(f"inference_time_batch_{batch_size}", time)

    def _measure_memory_usage(self, model) -> float:
        """Measure model's memory usage in MB."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        baseline = process.memory_info().rss / 1024 / 1024
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Get memory usage with model loaded
        with_model = process.memory_info().rss / 1024 / 1024
        
        return with_model - baseline

    def _measure_inference_speed(self, model, test_data: np.ndarray, 
                               batch_sizes: List[int]) -> List[float]:
        """Measure inference speed for different batch sizes."""
        inference_times = []
        
        for batch_size in batch_sizes:
            start_time = datetime.now()
            
            # Run predictions in batches
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i + batch_size]
                _ = model.predict(batch)
            
            end_time = datetime.now()
            inference_time = (end_time - start_time).total_seconds()
            inference_times.append(inference_time)
        
        return inference_times

    def _version_model(self, model_path: str, metrics: Dict[str, float]) -> str:
        """Version the model and save metrics."""
        # Generate version string
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        accuracy = metrics.get('accuracy', 0)
        version = f"v{timestamp}_acc{accuracy:.3f}"
        
        # Save model with version
        versioned_path = model_path.replace(
            os.path.splitext(model_path)[0],
            f"{os.path.splitext(model_path)[0]}_{version}"
        )
        os.rename(model_path, versioned_path)
        
        # Save metrics
        metrics_path = f"{os.path.splitext(versioned_path)[0]}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return version

    def _prepare_deployment(self, model, model_type: str, version: str):
        """Prepare model for deployment."""
        deployment_dir = os.path.join("deployment", version)
        os.makedirs(deployment_dir, exist_ok=True)
        
        # Save deployment config
        config = {
            'model_type': model_type,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'environment': self.config['deployment']['environment'],
            'requirements': self._get_requirements(model_type)
        }
        
        with open(os.path.join(deployment_dir, 'deployment_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # Generate API documentation
        self._generate_api_docs(deployment_dir, model_type)
        
        # Create deployment scripts
        self._create_deployment_scripts(deployment_dir, model_type)
        
        logging.info(f"Deployment artifacts prepared in {deployment_dir}")

    def _get_requirements(self, model_type: str) -> List[str]:
        """Get required packages for deployment."""
        common_requirements = [
            'numpy',
            'pandas',
            'pyyaml',
            'flask'
        ]
        
        if model_type == 'image':
            return common_requirements + [
                'tensorflow>=2.0.0',
                'pillow'
            ]
        else:
            return common_requirements + [
                'scikit-learn',
                'xgboost'
            ]

    def _generate_api_docs(self, deployment_dir: str, model_type: str):
        """Generate API documentation."""
        api_docs = {
            'endpoints': [
                {
                    'path': '/predict',
                    'method': 'POST',
                    'description': f'Make predictions with the {model_type} model',
                    'input_format': self._get_input_format(model_type),
                    'output_format': self._get_output_format(model_type)
                }
            ]
        }
        
        with open(os.path.join(deployment_dir, 'api_docs.json'), 'w') as f:
            json.dump(api_docs, f, indent=4)

    def _get_input_format(self, model_type: str) -> Dict:
        """Get input format specification."""
        if model_type == 'image':
            return {
                'type': 'object',
                'properties': {
                    'image': {
                        'type': 'string',
                        'format': 'base64',
                        'description': 'Base64 encoded image'
                    }
                }
            }
        else:
            return {
                'type': 'object',
                'properties': {
                    'features': {
                        'type': 'array',
                        'items': {
                            'type': 'number'
                        }
                    }
                }
            }

    def _create_deployment_scripts(self, deployment_dir: str, model_type: str):
        """Create deployment scripts."""
        # Create Flask app
        self._create_flask_app(deployment_dir, model_type)
        
        # Create Docker files
        self._create_docker_files(deployment_dir, model_type)
        
        # Create deployment instructions
        self._create_deployment_instructions(deployment_dir)

