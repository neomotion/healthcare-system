# multivariate_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MultivariateAnalysis:
    """
    A static class for performing multivariate data analysis.
    """

    @staticmethod
    def plot_pairplot(df, columns):
        """Plots a pair plot for the specified columns."""
        sns.pairplot(df[columns])
        plt.suptitle("Pair Plot of Selected Columns", y=1.02)
        plt.show()

    @staticmethod
    def plot_correlation_matrix(df, columns):
        """Plots a heatmap of the correlation matrix for the specified columns."""
        correlation_matrix = df[columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.show()

    @staticmethod
    def analyze_multivariate(df, columns):
        """Perform a full multivariate analysis including pair plots and correlation matrices."""
        MultivariateAnalysis.plot_pairplot(df, columns)
        MultivariateAnalysis.plot_correlation_matrix(df, columns)

