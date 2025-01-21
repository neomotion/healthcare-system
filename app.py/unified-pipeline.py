# unified_training_pipeline.py

import os
import logging
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

# Import your existing classes
from model_building import ModelBuilder
from data_preprocessing import DataPreprocessor
#from model_building_image import ImageClassificationModel


class UnifiedModelPipeline:
    def __init__(self):
        """Initialize the unified pipeline with logging configuration."""
        self.setup_logging()

    def setup_logging(self):
        """Configure logging with timestamp-based filename."""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Setup logging with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = logs_dir / f"model_training_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def detect_data_type(self, data_path: str) -> str:
        """
        Detect whether the input data is tabular or image based on the path.

        Returns:
        --------
        str
            'tabular' or 'image'
        """
        try:
            if os.path.isfile(data_path):
                # Check if it's a CSV or similar tabular file
                extension = os.path.splitext(data_path)[1].lower()
                if extension in ['.csv', '.xlsx', '.parquet']:
                    return 'tabular'

            elif os.path.isdir(data_path):
                # Check if directory contains image files
                valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
                for root, _, files in os.walk(data_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in valid_extensions):
                            return 'image'

            raise ValueError(f"Unable to determine data type for path: {data_path}")

        except Exception as e:
            self.logger.error(f"Error detecting data type: {str(e)}")
            raise

    def train_model(self) -> Dict[str, Any]:
        """
        Main pipeline that handles both tabular and image data.
        """
        try:
            # Get initial data path
            data_path = input("Enter the path to your dataset: ")
            data_type = self.detect_data_type(data_path)

            self.logger.info("Starting model training pipeline")
            self.logger.info(f"Data type: {data_type}")
            self.logger.info(f"Data path: {data_path}")

            # Create output directory if it doesn't exist
            output_dir = Path("models")
            output_dir.mkdir(exist_ok=True)

            if data_type == 'tabular':
                self.logger.info("Running tabular data pipeline")
                pipeline = ModelBuilder()
                # The train_and_evaluate_pipeline will handle its own input collection
                results = pipeline.train_and_evaluate_pipeline()

            else:  # image
                self.logger.info("Running image data pipeline")
                # Use the image pipeline's get_user_inputs
                pipeline = ImageClassificationModel()
                results = pipeline.image_model_pipeline()

            # Log results
            self.logger.info("Training completed successfully")
            self.logger.info(
                f"Model accuracy: {results.get('accuracy', results.get('training_metrics', {}).get('accuracy')) * 100:.2f}%")
            self.logger.info(
                f"Model recall: {results.get('recall', results.get('training_metrics', {}).get('recall')) * 100:.2f}%")
            self.logger.info(
                f"Model saved at: {results.get('model_path', results.get('model_info', {}).get('model_path'))}")

            return results

        except Exception as e:
            self.logger.error(f"Error in training pipeline: {str(e)}")
            raise


def main():
    """Main entry point for the pipeline."""
    try:
        pipeline = UnifiedModelPipeline()
        results = pipeline.train_model()
        print("\nTraining completed successfully!")
        print(f"Check the logs directory for detailed training information.")
        return results

    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        print("Check the logs directory for error details.")
        return None


if __name__ == "__main__":
    main()