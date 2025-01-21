# image_data_analysis.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

class ImageDataAnalyzer:
    """
    A static class for performing analysis on image datasets.
    """

    @staticmethod
    def load_image(image_path):
        """Loads an image from the specified path."""
        return Image.open(image_path)

    @staticmethod
    def display_image(image, title='Image'):
        """Displays an image."""
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()

    @staticmethod
    def analyze_image_data(image_dir):
        """
        Analyzes a directory of images by gathering dimensions and plotting distribution.
        
        Args:
            image_dir (str): Path to the directory containing images.
        """
        dimensions = []
        
        for image_file in os.listdir(image_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, image_file)
                with Image.open(image_path) as img:
                    dimensions.append(img.size)  # (width, height)

        # Convert dimensions to DataFrame for analysis
        df_dimensions = pd.DataFrame(dimensions, columns=['Width', 'Height'])

        # Display basic statistics
        print("Basic Statistics of Image Dimensions:")
        print(df_dimensions.describe())

        # Plot distribution of image dimensions
        plt.figure(figsize=(12, 6))
        sns.histplot(df_dimensions['Width'], bins=30, color='blue', kde=True, label='Width', stat='density', alpha=0.5)
        sns.histplot(df_dimensions['Height'], bins=30, color='orange', kde=True, label='Height', stat='density', alpha=0.5)
        plt.title('Distribution of Image Width and Height')
        plt.xlabel('Pixels')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def calculate_aspect_ratios(image_dir):
        """
        Calculates and returns aspect ratios for all images in the specified directory.
        
        Args:
            image_dir (str): Path to the directory containing images.
        
        Returns:
            list: A list of aspect ratios.
        """
        aspect_ratios = []

        for image_file in os.listdir(image_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, image_file)
                with Image.open(image_path) as img:
                    width, height = img.size
                    aspect_ratio = width / height
                    aspect_ratios.append(aspect_ratio)

        return aspect_ratios

    @staticmethod
    def plot_aspect_ratios(image_dir):
        """Plots a histogram of aspect ratios for images in the specified directory."""
        aspect_ratios = ImageDataAnalyzer.calculate_aspect_ratios(image_dir)

        plt.figure(figsize=(10, 6))
        sns.histplot(aspect_ratios, bins=30, kde=True)
        plt.title('Distribution of Image Aspect Ratios')
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # Example usage
    image_directory = "path/to/your/image/directory"  # Replace with your image directory
    analyzer = ImageDataAnalyzer()
    analyzer.analyze_image_data(image_directory)
    analyzer.plot_aspect_ratios(image_directory)

