import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
import os
import cv2

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data():
    # Load CIFAR-10 dataset
    (X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()
    
    # Select a subset: 5,000 training, 1,000 test
    train_subset_size = 5000
    test_subset_size = 1000
    indices_train = np.random.choice(X_train_full.shape[0], train_subset_size, replace=False)
    indices_test = np.random.choice(X_test_full.shape[0], test_subset_size, replace=False)
    X_train_full = X_train_full[indices_train]
    y_train_full = y_train_full[indices_train]
    X_test = X_test_full[indices_test]
    y_test = y_test_full[indices_test]
    
    # Resize images to 96x96 (minimum for MobileNetV2)
    def resize_images(images):
        resized = np.zeros((images.shape[0], 96, 96, 3), dtype=np.float32)
        for i in range(images.shape[0]):
            resized[i] = cv2.resize(images[i], (96, 96), interpolation=cv2.INTER_LINEAR)
        return resized
    
    X_train_full = resize_images(X_train_full)
    X_test = resize_images(X_test)
    
    # Apply MobileNetV2 preprocessing (scales to [-1, 1])
    X_train_full = preprocess_input(X_train_full)
    X_test = preprocess_input(X_test)
    
    # One-hot encode labels (10 classes)
    y_train_full = to_categorical(y_train_full, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    # Split train_full into train and validation (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_data_generators(X_train, y_train, X_val, y_val):
    # Simplified augmentation for training (only flip)
    train_datagen = ImageDataGenerator(horizontal_flip=True)
    
    # No augmentation for val/test
    val_datagen = ImageDataGenerator()
    
    # Generators
    batch_size = 16
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    
    return train_generator, val_generator

if __name__ == "__main__":
    # For testing
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    train_gen, val_gen = create_data_generators(X_train, y_train, X_val, y_val)
    os.makedirs('data', exist_ok=True)
    np.savez('data/preprocessed_data.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)