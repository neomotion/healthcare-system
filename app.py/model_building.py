# model_building.py

import xgboost as xgb
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import pandas as pd
from data_preprocessing import DataPreprocessor
from outlier_detection import OutlierAnalysis
from sklearn.metrics import accuracy_score, classification_report, recall_score
import joblib

class ClassificationType(Enum):
    """Enumeration for classification types"""
    BINARY = "binary"
    MULTI = "multi"

class ModelBuilder:
    """Static utility class for XGBoost model building and training"""
    
    @staticmethod
    def create_binary_classifier() -> xgb.XGBClassifier:
        """
        Creates a binary classification XGBoost model.
        
        Returns:
        --------
        xgb.XGBClassifier
            Configured binary classifier
        """
        return xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            verbosity=1,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    @staticmethod
    def create_multi_classifier(num_classes: int) -> xgb.XGBClassifier:
        """
        Creates a multiclass classification XGBoost model.
        
        Parameters:
        -----------
        num_classes : int
            Number of classes for classification
            
        Returns:
        --------
        xgb.XGBClassifier
            Configured multiclass classifier
            
        Raises:
        -------
        ValueError
            If num_classes <= 2
        """
        if num_classes <= 2:
            raise ValueError("Number of classes must be > 2 for multiclass classification")
        
        return xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=num_classes,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            verbosity=1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

    @staticmethod
    def build_model(classification_type: str, num_classes: Optional[int] = None) -> xgb.XGBClassifier:
        """
        Builds an XGBoost model based on the classification type.
        
        Parameters:
        -----------
        classification_type : str
            Type of classification ('binary' or 'multi')
        num_classes : int, optional
            Number of classes (required for multiclass classification)
            
        Returns:
        --------
        xgb.XGBClassifier
            Configured XGBoost classifier
            
        Raises:
        -------
        ValueError
            If invalid classification type or missing num_classes for multiclass
        """
        try:
            clf_type = ClassificationType(classification_type.lower())
        except ValueError:
            raise ValueError(f"Invalid classification type. Must be one of: {[t.value for t in ClassificationType]}")
        
        if clf_type == ClassificationType.BINARY:
            return ModelBuilder.create_binary_classifier()
        else:
            return ModelBuilder.create_multi_classifier(num_classes)

    @staticmethod
    def prepare_data(data_path: str, target_column: str):
        """
        Prepares data for model training.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset file
        target_column : str
            Name of the target column
        outlier_column : str
            Column to use for outlier detection
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            Training and test splits (X_train, X_test, y_train, y_test)
        """
        return DataPreprocessor.preprocessing_pipeline(data_path, target_column)

    @staticmethod
    def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluates the model and returns performance metrics.

        Parameters:
        -----------
        model : xgb.XGBClassifier
            Trained XGBoost model
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            True test labels

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing accuracy, recall, and classification report
        """
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')  # Using macro average for multiclass
        report = classification_report(y_test, y_pred)
        return {
            'accuracy': accuracy,
            'recall': recall,
            'report': report
        }

    @staticmethod
    def save_model(model: xgb.XGBClassifier, model_name: str) -> str:
        """
        Saves the model to disk.
        
        Parameters:
        -----------
        model : xgb.XGBClassifier
            Trained model to save
        model_name : str
            Name for the saved model file
            
        Returns:
        --------
        str
            Path to the saved model file
        """
        model_path = f"models/{model_name}.joblib"
        joblib.dump(model, model_path)
        return model_path

    @staticmethod
    def get_user_inputs() -> Tuple[str, str, str]:
        """
        Gets and validates user inputs for model training.

        Returns:
        --------
        Tuple[str, str, str, str]
            Tuple containing validated data_path, target_column, outlier_column, and classification_type
        """
        data_file_path = input("Enter the path to your dataset: ")
        target_col = input("Enter the target column name: ")
        model_filename = input("Enter the name to save the model (without extension): ")

        return data_file_path, target_col, model_filename

    @staticmethod
    def train_and_evaluate_pipeline() -> Dict[str, Any]:
        """
        Complete pipeline for training and evaluating an XGBoost model.
        """
        # User inputs
        data_path, target_column, model_name = ModelBuilder.get_user_inputs()

        # Prepare data
        X_train, X_test, y_train, y_test = ModelBuilder.prepare_data(
            data_path, target_column
        )
        #print(y_train.head())

        # Determine number of classes and build model
        #if not isinstance(y_train, pd.Series):
            #raise ValueError("y_train must be a pandas Series.")
        # Assuming y_train is a pandas Series
        y = y_train.copy()
        y = y.squeeze()
        num_unique = y.nunique()
        if num_unique <= 2:
            classification_type = 'binary'
        else:
            classification_type = 'multi'

        model = ModelBuilder.build_model(
            classification_type=classification_type,
            num_classes=num_unique if classification_type.lower() == 'multi' else None
        )

        """model = ModelBuilder.build_model(
            classification_type="binary",
            num_classes=2
        )"""

        # Train model
        model.fit(X_train, y_train, verbose=10)

        # Evaluate model
        metrics = ModelBuilder.evaluate_model(model, X_test, y_test)

        # Save model
        model_path = ModelBuilder.save_model(model, model_name)

        # Print results
        print(f"Model Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"Model Recall: {metrics['recall'] * 100:.2f}%")
        print("\nClassification Report:")
        print(metrics['report'])
        print(f"\nModel saved as {model_path}")

        return {
            'accuracy': metrics['accuracy'],
            'recall': metrics['recall'],
            'report': metrics['report'],
            'model_path': model_path
        }
def main():

    ModelBuilder.train_and_evaluate_pipeline()


if __name__ == "__main__":
    main()





#/home/sahilsssingh5/healthcare0.1/data/hepatitis/hepatitis.csv
















