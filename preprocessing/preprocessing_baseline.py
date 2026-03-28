import cv2

# Resizing the image
def preprocess_baseline(image):
    return cv2.resize(image, (640, 640))