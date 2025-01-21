import os
import numpy as np
import cv2
from typing import Tuple,Dict
import tensorflow as tf
#from tensorflow import sklearn
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2, ResNet50
from abc import ABC, abstractmethod

class ModelStrategy(ABC):
    """Abstract base class for model architecture strategies"""
    
    @abstractmethod
    def create_model(self, input_shape, num_classes):
        """Creates and returns a model with the specified input shape and number of classes"""
        pass

class ClassificationStrategy(ABC):
    """Abstract base class for classification strategies"""
    
    @abstractmethod
    def get_final_activation(self):
        """Returns the activation function for the final layer"""
        pass
    
    @abstractmethod
    def get_loss_function(self):
        """Returns the loss function for the model"""
        pass
    
    @abstractmethod
    def get_metrics(self):
        """Returns the metrics for model evaluation"""
        pass
    
    @abstractmethod
    def process_labels(self, labels):
        """Process labels according to classification type"""
        pass

class BinaryClassificationStrategy(ClassificationStrategy):
    """Strategy for binary classification"""
    @abstractmethod
    def get_final_activation(self):
        return 'sigmoid'

    @abstractmethod
    def get_loss_function(self):
        return 'binary_crossentropy'

    @abstractmethod
    def get_metrics(self):
        return ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

    @abstractmethod
    def process_labels(self, labels):
        # Convert labels to binary format
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        return lb.fit_transform(labels).ravel()

class MultiClassificationStrategy(ClassificationStrategy):
    """Strategy for multiclass classification"""

    @abstractmethod
    def get_final_activation(self):
        return 'softmax'

    @abstractmethod
    def get_loss_function(self):
        return 'sparse_categorical_crossentropy'

    @abstractmethod
    def get_metrics(self):
        return ['accuracy']

    @abstractmethod
    def process_labels(self, labels):
        # Convert string labels to numeric indices
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        return le.fit_transform(labels)

class CustomCNNStrategy(ModelStrategy):
    """Strategy for creating a custom CNN architecture"""

    @abstractmethod
    def __init__(self, classification_strategy):
        self.classification_strategy = classification_strategy

    @abstractmethod
    def create_model(self, input_shape, num_classes):
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(num_classes if num_classes > 2 else 1, 
                  activation=self.classification_strategy.get_final_activation())
        ])
        return model


class MobileNetStrategy(ModelStrategy):
    """Strategy for using MobileNetV2 architecture"""

    @abstractmethod
    def __init__(self, classification_strategy):
        self.classification_strategy = classification_strategy

    @abstractmethod
    def create_model(self, input_shape, num_classes):
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes if num_classes > 2 else 1,
                       activation=self.classification_strategy.get_final_activation())(x)
        
        model = Model(base_model.input, outputs)
        return model

class ResNetStrategy(ModelStrategy):
    """Strategy for using ResNet50 architecture"""

    @abstractmethod
    def __init__(self, classification_strategy):
        self.classification_strategy = classification_strategy

    @abstractmethod
    def create_model(self, input_shape, num_classes):
        base_model = ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes if num_classes > 2 else 1,
                       activation=self.classification_strategy.get_final_activation())(x)
        
        model = Model(base_model.input, outputs)
        return model

class ImageClassificationModel:
    """
    A class for building and training image classification models with different architectures
    and classification strategies.
    """

    @abstractmethod
    def __init__(self, model_strategy='custom', classification_type='multiclass', 
                 img_size=(224, 224), batch_size=32, epochs=20):
        """
        Initialize the image classification model.
        
        Args:
            model_strategy (str): Type of model to use ('custom', 'mobilenet', or 'resnet')
            classification_type (str): Type of classification ('binary' or 'multiclass')
            img_size (tuple): Input image dimensions
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.set_classification_strategy(classification_type)
        self.set_model_strategy(model_strategy)
        self.label_encoder = None

    @abstractmethod
    def set_classification_strategy(self, classification_type):
        """Set the classification strategy"""
        strategies = {
            'binary': BinaryClassificationStrategy(),
            'multiclass': MultiClassificationStrategy()
        }
        
        if classification_type not in strategies:
            raise ValueError(f"Invalid classification type. Choose from: {list(strategies.keys())}")
        
        self.classification_strategy = strategies[classification_type]

    @abstractmethod
    def set_model_strategy(self, strategy_name):
        """Set the model architecture strategy"""
        strategies = {
            'custom': CustomCNNStrategy(self.classification_strategy),
            'mobilenet': MobileNetStrategy(self.classification_strategy),
            'resnet': ResNetStrategy(self.classification_strategy)
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Invalid strategy name. Choose from: {list(strategies.keys())}")
        
        self.model_strategy = strategies[strategy_name]

    @abstractmethod
    def load_data(self, image_dir, labels):
        """
        Loads images from a directory and prepares data for training.
        """
        images = []
        for label in labels:
            image_path = os.path.join(image_dir, label)
            img = cv2.imread(image_path)
            img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        
        # Process labels according to classification strategy
        processed_labels = self.classification_strategy.process_labels(labels)
        
        return np.array(images), processed_labels

    @abstractmethod
    def preprocess_data(self, X, y):
        """Preprocesses the data by normalizing the pixel values."""
        X = X.astype('float32') / 255.0
        return X, y

    @abstractmethod
    def train_model(self, X, y):
        """
        Trains the selected model architecture and returns training metrics.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing training history, model performance metrics,
            and evaluation results
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Data augmentation
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            rotation_range=20,
            zoom_range=0.2,
            shear_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2
        )
        datagen.fit(X_train)

        # Create model using the selected strategy
        num_classes = len(np.unique(y))
        self.model = self.model_strategy.create_model(
            input_shape=(self.img_size[0], self.img_size[1], 3),
            num_classes=num_classes
        )

        # Compile model with strategy-specific parameters
        self.model.compile(
            optimizer='adam',
            loss=self.classification_strategy.get_loss_function(),
            metrics=self.classification_strategy.get_metrics()
        )

        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            callbacks=[early_stopping]
        )

        # Evaluate model
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred_classes)
        recall = recall_score(y_val, y_pred_classes, average='macro')
        report = classification_report(y_val, y_pred_classes)

        # Print evaluation results
        print("\nModel Evaluation Results:")
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        print(f"Validation Recall: {recall * 100:.2f}%")
        print("\nClassification Report:")
        print(report)

        return {
            'history': history.history,
            'accuracy': accuracy,
            'recall': recall,
            'report': report,
            'model': self.model,
            'final_epochs': len(history.history['loss'])
        }

    @abstractmethod
    def save_model(self, model_name):
        """Saves the trained model to a file."""
        self.model.save(model_name)

    @abstractmethod
    def load_model(self, model_path):
        """Loads a saved model from a file."""
        self.model = tf.python.keras.models.load_model(model_path)

    @abstractmethod
    def get_user_inputs(self) -> Tuple[str, str, str, str, dict]:
        """
        Gets and validates user inputs for model training, with automatic label mapping
        from folder structure.

        Returns:
        --------
        Tuple[str, str, str, str, dict]
            Tuple containing validated data_path, model_strategy, classification_type,
            model_name, and label_mapping
        """
        # Get basic inputs
        data_file_path = input("Enter the path to your dataset: ")

        # Validate folder structure
        test_path = os.path.join(data_file_path, "test")
        train_path = os.path.join(data_file_path, "train")

        if not (os.path.exists(test_path) and os.path.exists(train_path)):
            raise ValueError("Dataset must contain 'test' and 'train' folders")

        # Get class names from test folder structure
        class_names = [d for d in os.listdir(test_path)
                       if os.path.isdir(os.path.join(test_path, d))]

        if not class_names:
            raise ValueError("No class folders found in test directory")

        # Create label mapping automatically
        label_mapping = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}

        # Determine classification type based on number of classes
        classification_type = 'binary' if len(class_names) == 2 else 'multiclass'

        # Get remaining inputs
        model_strategy = input("Enter the model strategy (custom, mobilenet, resnet): ")
        model_name = input("Enter the name to save the model (without extension): ")

        # Validate inputs
        print("\nConfirming your inputs:")
        print(f"Dataset path: {data_file_path}")
        print(f"Model strategy: {model_strategy}")
        print(f"Classification type: {classification_type} ({len(class_names)} classes detected)")
        print(f"Model name: {model_name}")
        print("Detected classes and label mapping:")
        for class_name, label in label_mapping.items():
            print(f"  {class_name}: {label}")

        confirm = input("\nIs this correct? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Please start over with correct inputs.")
            return self.get_user_inputs()

        return data_file_path, model_strategy, classification_type, model_name, label_mapping

    @abstractmethod
    def image_model_pipeline(self):
        """
        Runs the complete image classification pipeline and returns model metrics.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing model configuration, training results,
            and performance metrics
        """
        # Get user inputs
        base_directory, model_strategy, classification_type, model_name, label_mapping = self.get_user_inputs()

        # Validate model name and add extension
        model_path = f"{model_name}.h5" if not model_name.endswith('.h5') else model_name

        # Create model with desired architecture and classification type
        img_classifier = ImageClassificationModel(
            model_strategy=model_strategy,
            classification_type=classification_type,
            img_size=(224, 224),
            batch_size=32,
            epochs=20
        )

        # Load and preprocess data from train directory
        train_dir = os.path.join(base_directory, "train")
        X_train, y_train = img_classifier.load_data(train_dir, label_mapping)
        X_train, y_train = img_classifier.preprocess_data(X_train, y_train)

        # Load and preprocess data from test directory
        test_dir = os.path.join(base_directory, "test")
        X_test, y_test = img_classifier.load_data(test_dir, label_mapping)
        X_test, y_test = img_classifier.preprocess_data(X_test, y_test)

        # Train model
        training_results = img_classifier.train_model(X_train, y_train)

        # Evaluate on test set and update training_results
        y_pred = training_results['model'].predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

        # Update training_results with test set metrics
        training_results['accuracy'] = accuracy_score(y_test_classes, y_pred_classes)
        training_results['recall'] = recall_score(y_test_classes, y_pred_classes, average='weighted')
        training_results['report'] = classification_report(y_test_classes, y_pred_classes)

        # Save model
        img_classifier.save_model(model_path)
        print(f"Model training complete and saved as {model_path}")

        return {
            'model_config': {
                'model_strategy': model_strategy,
                'classification_type': classification_type,
                'img_size': (224, 224),
                'batch_size': 32,
                'epochs': 20
            },
            'training_metrics': {
                'history': training_results['history'],
                'final_epochs': training_results['final_epochs'],
                'accuracy': training_results['accuracy'],
                'recall': training_results['recall']
            },
            'evaluation': {
                'classification_report': training_results['report']
            },
            'model_info': {
                'model': training_results['model'],
                'model_path': model_path,
                'label_mapping': label_mapping
            },
            'data_info': {
                'input_shape': X_train.shape,
                'num_classes': len(label_mapping),
                'dataset_path': base_directory
            }
        }