#data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class DataPreprocessor:
    """
    A static class for data preprocessing tasks like loading, cleaning, preprocessing, and splitting data.
    """

    @staticmethod
    def load_data(file_path):
        """Load data from a CSV file."""
        return pd.read_csv(file_path)

    @staticmethod
    def clean_data(df):
        """Clean the dataset by dropping rows with null values."""
        # Drop the 'Unnamed: 32' column
        if 'Unnamed: 32' in df.columns:
            df = df.drop(columns=['Unnamed: 32'])
            print("Dropped 'Unnamed: 32' column.")

        print("Null value counts:\n", df.isnull().sum())
        df_cleaned = df.dropna()
        print("Data after cleaning:\n", df_cleaned.head())
        return df_cleaned

    @staticmethod
    def categorical_encoding(data):
        """
        Encodes all categorical columns in the DataFrame with integer values (1, 2, 3, ...).

        Parameters:
        data (pd.DataFrame): A DataFrame containing only categorical columns.

        Returns:
        pd.DataFrame: The DataFrame with encoded values in the same columns.
        """
        for col in data.columns:
            data[col], _ = pd.factorize(data[col], sort=True)
            """data[col] += 1  # To start encoding from 1 instead of 0"""
        return data

    @staticmethod
    def feature_selection(data, target_column):
        """
        Trains a Random Forest model, calculates feature importance, and removes unimportant features.

        Parameters:
        data (pd.DataFrame): The complete dataset with features and target column.
        target_column (str): The name of the target column in the dataset.
        threshold (float): The minimum importance score to retain a feature. Default is 0.01.

        Returns:
        pd.DataFrame: Dataset with only important features retained.
        """
        threshold = 0.01
        # Split features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Initialize the Random Forest model
        model = RandomForestClassifier(random_state=42)

        # Train the model
        model.fit(X, y)

        # Get feature importance
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)

        # Filter features based on the threshold
        important_features = feature_importance[feature_importance > threshold].index

        # Keep only important features in the dataset
        X_important = X[important_features]

        # Add the target column back to the dataset
        X_important[target_column] = y

        # Print feature importance scores
        print("Feature Importance Scores:")
        print(feature_importance.sort_values(ascending=False))

        print("\nRetained Features:")
        print(important_features.tolist())

        return X_important

    @staticmethod
    def preprocess_features(df, target_column):
        """
        Preprocess features: Separate categorical and numerical data,
        one-hot encode categorical features, and join them back into a single DataFrame.
        """
        # Convert to DataFrame if it's a Series
        if isinstance(df, pd.Series):
            df = df.to_frame()
            print(df.head())

        # Validate input is now a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame or Series")

        #df = DataPreprocessor.feature_selection(df,target_column)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()


        # Remove target column from numerical columns if it's included
        #numerical_cols.remove(target_column) if target_column in numerical_cols else None

        # Retain numerical columns without scaling
        df_numerical = df[numerical_cols]

        # One-hot encode categorical columns
        """df_categorical = pd.get_dummies(df[categorical_cols], drop_first=True)"""
        df_categorical = DataPreprocessor.categorical_encoding(df[categorical_cols])

        # Combine numerical and categorical features into one DataFrame
        df_processed = pd.concat([df_numerical, df_categorical], axis=1)

        return df_processed

    @staticmethod
    def split_data(df, target_column):
        """Split the dataset into training and test sets."""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    @staticmethod
    def preprocessing_pipeline(file_path, target_column):
        """
        Full pipeline to load, clean, preprocess, and split the data.

        Args:
            file_path (str): Path to the CSV file.
            target_column (str): Name of the target column.

        Returns:
            tuple: Processed training and test sets (X_train, X_test, y_train, y_test).
        """
        print("Loading data...")
        data = DataPreprocessor.load_data(file_path)

        print("Cleaning data...")
        cleaned_data = DataPreprocessor.clean_data(data)

        print("Splitting data...")
        X_train, X_test, y_train, y_test = DataPreprocessor.split_data(cleaned_data,target_column)
        print(y_train.head())

        print("Preprocessing training data...")
        X_train = DataPreprocessor.preprocess_features(X_train, target_column)
        y_train = DataPreprocessor.preprocess_features(y_train,target_column)
        print(y_train.head())

        print("Preprocessing test data...")
        X_test = DataPreprocessor.preprocess_features(X_test, target_column)
        y_test = DataPreprocessor.preprocess_features(y_test,target_column)

        print("Pre-processing Pipeline completed successfully.")
        print(y_train.head())
        print(X_train.head())

        return X_train, X_test, y_train, y_test



