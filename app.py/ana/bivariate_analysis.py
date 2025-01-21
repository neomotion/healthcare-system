# bivariate_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BivariateAnalysis:
    """
    A static class for performing bivariate data analysis.
    """

    @staticmethod
    def plot_scatter(df, x_column, y_column):
        """Plots a scatter plot between two columns."""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[x_column], y=df[y_column])
        plt.title(f'Scatter plot of {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid()
        plt.show()

    @staticmethod
    def plot_correlation(df, x_column, y_column):
        """Calculates and prints the correlation between two columns."""
        correlation = df[[x_column, y_column]].corr().iloc[0, 1]
        print(f'Correlation between {x_column} and {y_column}: {correlation:.2f}')
        return correlation

    @staticmethod
    def analyze_bivariate(df, x_column, y_column):
        """Perform a full bivariate analysis including a scatter plot and correlation."""
        BivariateAnalysis.plot_scatter(df, x_column, y_column)
        BivariateAnalysis.plot_correlation(df, x_column, y_column)

