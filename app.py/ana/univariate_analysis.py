# univariate_analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateAnalysis:
    """
    A static class for performing univariate data analysis.
    """

    @staticmethod
    def describe_data(df):
        """Provides basic statistics for the DataFrame."""
        return df.describe()

    @staticmethod
    def plot_histogram(df, column):
        """Plots a histogram for the specified column."""
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(axis='y')
        plt.show()

    @staticmethod
    def plot_boxplot(df, column):
        """Plots a boxplot for the specified column."""
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.show()

    @staticmethod
    def analyze_univariate(df, column):
        """Perform a full univariate analysis on a column."""
        stats = UnivariateAnalysis.describe_data(df[[column]])
        print(f'Descriptive Statistics for {column}:\n', stats)
        UnivariateAnalysis.plot_histogram(df, column)
        UnivariateAnalysis.plot_boxplot(df, column)

