#outlier_detection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class OutlierAnalysis:
    """
    A static class for detecting and handling outliers in a dataset.
    """

    @staticmethod
    def detect_outliers_boxplot(df, column):
        """Detect outliers using the boxplot method."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers

    @staticmethod
    def visualize_outliers(df, column):
        """Visualize outliers using a boxplot."""
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column} with Outliers')
        plt.show()

    @staticmethod
    def analyze_outliers(df, column):
        """Perform outlier analysis on a column and visualize the results."""
        outliers = OutlierAnalysis.detect_outliers_boxplot(df, column)
        print(f"Outliers detected in {column}:\n", outliers)
        OutlierAnalysis.visualize_outliers(df, column)
        return df.drop(outliers.index), outliers



    #Multiple Detection Methods:
    #   IQR method (using quartiles)
    #   Z-score method (using standard deviations)
    #Multiple Handling Strategies:
    #    Remove: Completely removes rows with outliers
    #    Clip: Caps values at the bounds
    #    Fill: Replaces outliers with specified value
    #Flexible Fill Options:
    #    Mean
    #    Median
    #    Custom numeric value
    #Support for Multiple Columns:
    #    Can process single column or multiple columns
    #    Maintains separate outlier information for each column
    @staticmethod
    def remove_outliers(df, columns, method='iqr', threshold=1.5, strategy='remove', fill_value=None):
        """
        Remove or handle outliers in specified columns using various methods.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame containing the data
        columns : str or list
            Column name(s) to check for outliers
        method : str, optional (default='iqr')
            Method to detect outliers: 'iqr' or 'zscore'
        threshold : float, optional (default=1.5)
            Threshold for outlier detection
            - For IQR method: multiply IQR by this value
            - For Z-score method: number of standard deviations
        strategy : str, optional (default='remove')
            Strategy to handle outliers: 'remove', 'clip', or 'fill'
        fill_value : str or float, optional (default=None)
            If strategy is 'fill', value to replace outliers with
            Can be 'mean', 'median', or a numeric value
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with handled outliers
        dict
            Dictionary containing outlier indices and values for each column
        """
        if isinstance(columns, str):
            columns = [columns]
            
        df_clean = df.copy()
        outliers_info = {}
        
        for column in columns:
            if method == 'iqr':
                # IQR method
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            
            elif method == 'zscore':
                # Z-score method
                z_scores = (df[column] - df[column].mean()) / df[column].std()
                outlier_mask = abs(z_scores) > threshold
            
            else:
                raise ValueError("Method must be either 'iqr' or 'zscore'")
            
            # Store outlier information
            outliers_info[column] = {
                'indices': df[outlier_mask].index.tolist(),
                'values': df[column][outlier_mask].tolist(),
                'count': outlier_mask.sum()
            }
            
            # Handle outliers based on chosen strategy
            if strategy == 'remove':
                df_clean = df_clean[~outlier_mask]
                
            elif strategy == 'clip':
                if method == 'iqr':
                    df_clean[column] = df_clean[column].clip(lower=lower_bound, upper=upper_bound)
                else:  # zscore
                    mean, std = df[column].mean(), df[column].std()
                    df_clean[column] = df_clean[column].clip(
                        lower=mean - threshold * std,
                        upper=mean + threshold * std
                    )
                    
            elif strategy == 'fill':
                if fill_value == 'mean':
                    replacement = df[column].mean()
                elif fill_value == 'median':
                    replacement = df[column].median()
                elif isinstance(fill_value, (int, float)):
                    replacement = fill_value
                else:
                    raise ValueError("fill_value must be 'mean', 'median', or a numeric value")
                
                df_clean.loc[outlier_mask, column] = replacement
                
            else:
                raise ValueError("Strategy must be 'remove', 'clip', or 'fill'")
            
            # Print summary
            print(f"\nOutlier Summary for {column}:")
            print(f"Number of outliers detected: {outliers_info[column]['count']}")
            print(f"Percentage of outliers: {(outliers_info[column]['count']/len(df)*100):.2f}%")
            
        return df_clean, outliers_info
    
