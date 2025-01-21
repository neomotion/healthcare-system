# image_processing.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    """
    A static class for image processing tasks, including resizing, normalization,
    data augmentation, and other preprocessing steps for deep learning pipelines.
    """

    @staticmethod
    def load_image(image_path):
        """
        Loads an image from the specified path in RGB format.
        """
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    @staticmethod
    def display_image(image, title='Image'):
        """
        Displays an image.
        """
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()

    @staticmethod
    def resize_image(image, width, height):
        """
        Resizes the image to the specified width and height.
        """
        return cv2.resize(image, (width, height))

    @staticmethod
    def normalize_image(image, mean, std):
        """
        Normalizes the image using the specified mean and standard deviation.
        Pixel values are scaled to [0, 1], then normalized.
        """
        image = image / 255.0  # Scale to [0, 1]
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        return (image - mean) / std

    @staticmethod
    def convert_to_grayscale(image):
        """
        Converts an image to grayscale.
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def apply_gaussian_blur(image, kernel_size=(5, 5)):
        """
        Applies Gaussian blur to the image.
        """
        return cv2.GaussianBlur(image, kernel_size, 0)

    @staticmethod
    def random_crop(image, crop_size):
        """
        Randomly crops an image to the specified size.
        """
        height, width, _ = image.shape
        crop_width, crop_height = crop_size
        if width < crop_width or height < crop_height:
            raise ValueError("Crop size must be smaller than the image dimensions.")
        x = np.random.randint(0, width - crop_width + 1)
        y = np.random.randint(0, height - crop_height + 1)
        return image[y:y+crop_height, x:x+crop_width]

    @staticmethod
    def horizontal_flip(image):
        """
        Flips the image horizontally with a 50% chance.
        """
        if np.random.rand() > 0.5:
            return cv2.flip(image, 1)
        return image

    @staticmethod
    def augment_image(image, crop_size=(224, 224)):
        """
        Applies a series of data augmentation techniques to the image.
        Includes random cropping and horizontal flipping.
        """
        image = ImageProcessor.random_crop(image, crop_size)
        image = ImageProcessor.horizontal_flip(image)
        return image

    @staticmethod
    def image_pipeline(image_path):
        """
        Complete preprocessing pipeline for a single image.
        - Load image
        - Resize to target size
        - Data augmentation (only for training)
        - Normalize using the specified mean and std
        """
        target_size = (224, 224),
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Load and resize
        image = ImageProcessor.load_image(image_path)
        image = ImageProcessor.resize_image(image, *target_size)

        # Normalize
        image = ImageProcessor.normalize_image(image, mean, std)

        return image
