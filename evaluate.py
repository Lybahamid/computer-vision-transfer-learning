import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_and_preprocess_data

def evaluate_model(model_path='models/best_model.keras'):
    # Load data (we only need test)
    _, _, _, _, X_test, y_test = load_and_preprocess_data()
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Predict on test
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Metrics
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    
    # Classification report
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    plt.show()
    
    # Visualize a few misclassified images
    mis_idx = np.where(y_pred_classes != y_true)[0][:9]
    if len(mis_idx) > 0:
        plt.figure(figsize=(10, 10))
        for i, idx in enumerate(mis_idx):
            plt.subplot(3, 3, i+1)
            plt.imshow((X_test[idx] + 1) / 2)  # Undo MobileNetV2 preprocessing for visualization
            plt.title(f"True: {class_names[y_true[idx]]}, Pred: {class_names[y_pred_classes[idx]]}")
            plt.axis('off')
        plt.savefig('models/misclassified.png')
        plt.show()

if __name__ == "__main__":
    evaluate_model()