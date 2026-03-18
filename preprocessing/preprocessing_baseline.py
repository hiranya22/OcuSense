import cv2

def preprocess_baseline(image):
    return cv2.resize(image, (640, 640))