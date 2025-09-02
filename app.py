import tensorflow as tf
import numpy as np
import cv2
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import uuid

app = FastAPI()

# Load the trained model
model = load_model('D:/computer-vision-transfer-learning/models/best_model.keras')

# Initialize the model with a dummy input to define outputs
dummy_input = tf.zeros((1, 96, 96, 3))
_ = model(dummy_input)

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define model_modifier for Grad-CAM
def model_modifier(m):
    base_model = m.get_layer('mobilenetv2_1.00_96')
    return tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('out_relu').output)

# Initialize Grad-CAM
gradcam = Gradcam(model, model_modifier=model_modifier, clone=True)

# Define loss function for Grad-CAM (sum over feature maps)
def loss(output):
    return tf.reduce_sum(output)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the uploaded image
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(content={"error": "Invalid image file."}, status_code=400)
    img_resized = cv2.resize(img, (96, 96))
    img_input = np.expand_dims(img_resized, axis=0)
    img_input = preprocess_input(img_input)  # MobileNetV2 preprocessing

    # Get model prediction
    pred = model.predict(img_input)
    pred_class = class_names[np.argmax(pred[0])]
    confidence = float(np.max(pred[0]))

    # Generate Grad-CAM heatmap
    cam = gradcam(loss, img_input)
    heatmap = normalize(cam[0])
    heatmap = cv2.resize(heatmap, (96, 96))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0.0)

    # Save heatmap image to a file
    filename = f"heatmap_{uuid.uuid4().hex}.png"
    filepath = os.path.join("D:/computer-vision-transfer-learning/heatmaps", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, superimposed_img)

    # Return response with filename
    return JSONResponse(content={
        "predicted_class": pred_class,
        "confidence": confidence,
        "heatmap_file": filename
    })

@app.get("/heatmap/{filename}")
async def get_heatmap(filename: str):
    filepath = os.path.join("D:/computer-vision-transfer-learning/heatmaps", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png")
    return JSONResponse(content={"error": "File not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)