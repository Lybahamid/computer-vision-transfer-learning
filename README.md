## Computer Vision Transfer Learning Project

Overview

This project implements a computer vision application using transfer learning with MobileNetV2. It includes a FastAPI backend that predicts image classes (e.g., airplane, automobile, etc.) and generates Grad-CAM heatmaps to visualize the areas of the image that influenced the prediction.

Features

Classifies images into 10 categories using a pre-trained MobileNetV2 model.
Generates Grad-CAM heatmaps to highlight important regions in the image.
Exposes a REST API for predictions and heatmap retrieval.
Saves heatmap images for later access.

Prerequisites

Python 3.10
Conda environment (recommended)
Required libraries: tensorflow, numpy, opencv-python, fastapi, uvicorn, tf-keras-vis

Setup Instructions

Clone the repository:git clone https://github.com/Lybahamid/computer-vision-transfer-learning.git

cd computer-vision-transfer-learning


Create a Conda environment:conda create -n lfw_env python=3.10
conda activate lfw_env


Install dependencies:
pip install tensorflow numpy opencv-python fastapi uvicorn tf-keras-vis


Place your trained model file (best_model.keras) in the models directory:
Path: D:/computer-vision-transfer-learning/models/best_model.keras


Create a heatmaps directory for storing generated heatmap images:
Path: D:/computer-vision-transfer-learning/heatmaps



Running the Application

Start the FastAPI server:
python app.py


Test the API with curl:
Upload an image for prediction:
curl -X POST "http://localhost:8000/predict" -F "file=@image1.jpg"


Download the generated heatmap:
curl -X GET "http://localhost:8000/heatmap/heatmap_123e4567-e89b-12d3-a456-426614174000.png" -o heatmap.png





API Endpoints

POST /predict: Upload an image to get a prediction and heatmap filename.
GET /heatmap/{filename}: Retrieve a saved heatmap image.

File Structure

app.py: Main FastAPI application code.
models/: Directory for the trained model file.
heatmaps/: Directory for storing generated heatmap images.

Contributing
Feel free to submit issues or pull requests. Please ensure your changes align with the project’s goals.
