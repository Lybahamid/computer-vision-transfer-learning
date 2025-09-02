import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from preprocess import load_and_preprocess_data
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('D:/computer-vision-transfer-learning/models/best_model.keras')

# Initialize the model with a dummy input to define outputs
dummy_input = tf.zeros((1, 96, 96, 3))
_ = model(dummy_input)

# Load and preprocess test data
_, _, _, _, X_test, y_test = load_and_preprocess_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Debug model summary to identify layers
model.summary()
base_model = model.get_layer('mobilenetv2_1.00_96')
print("Base model layers:")
for layer in base_model.layers:
    print(f"Layer name: {layer.name}")

# Define a model_modifier to set the last convolutional layer for Grad-CAM
def model_modifier(m):
    base_model = m.get_layer('mobilenetv2_1.00_96')
    return tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('out_relu').output)

# Initialize Grad-CAM
gradcam = Gradcam(model, model_modifier=model_modifier, clone=True)

# Define the loss function for Grad-CAM (sum over feature maps)
def loss(output):
    return tf.reduce_sum(output)

# Visualize Grad-CAM for a few test images
num_images = 5
plt.figure(figsize=(15, 3 * num_images))
for i in range(num_images):
    img = X_test[i:i+1]
    true_label = class_names[np.argmax(y_test[i])]
    pred = model.predict(img)
    pred_label = class_names[np.argmax(pred[0])]
    
    # Generate Grad-CAM heatmap
    cam = gradcam(loss, img)
    heatmap = normalize(cam[0])  # cam is a list/array, take the first element
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (96, 96))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.uint8((img[0] + 1) * 127.5), 0.6, heatmap, 0.4, 0.0)
    
    # Plot
    plt.subplot(num_images, 2, 2*i+1)
    plt.imshow((img[0] + 1) / 2)
    plt.title(f'True: {true_label}, Pred: {pred_label}')
    plt.axis('off')
    
    plt.subplot(num_images, 2, 2*i+2)
    plt.imshow(superimposed_img)
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    # ...existing code...
plt.tight_layout()
plt.savefig('models/grad_cam_visualization.png')
plt.show()