import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt
import os
from preprocess import load_and_preprocess_data, create_data_generators

# Set random seed
tf.random.set_seed(42)

def build_model(input_shape=(96, 96, 3), num_classes=10):
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom top layers
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(optimizer=Adam(learning_rate=0.0003),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(model, train_generator, val_generator, epochs=120):
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('models/best_model.keras', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    tqdm_callback = TqdmCallback(verbose=1)
    
    # Train
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping, checkpoint, reduce_lr, tqdm_callback],
        verbose=0
    )
    
    return history

def plot_history(history, title='Training History'):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    
    plt.savefig(f'models/{title.lower().replace(" ", "_")}.png')
    plt.show()

if __name__ == "__main__":
    # Load data and generators
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    train_gen, val_gen = create_data_generators(X_train, y_train, X_val, y_val)
    
    # Build and train
    model = build_model()
    history = train_model(model, train_gen, val_gen)
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    plot_history(history, 'Initial Training')
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    model.save('models/final_model.keras')