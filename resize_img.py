# filepath: d:\computer-vision-transfer-learning\resize_img.py
import cv2
img = cv2.imread('D:/computer-vision-transfer-learning/image.jpg')
if img is None:
    raise FileNotFoundError("Image not found at '/image.jpg'")
img = cv2.resize(img, (96, 96))
cv2.imwrite('D:/computer-vision-transfer-learning/image1.jpg', img)