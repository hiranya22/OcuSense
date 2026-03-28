import cv2
import numpy as np
# ----------------------- Pipeline A: Channel Stacking ----------------------- #

CLIP_LIMIT = 1.0
TILE_SIZE = (8, 8)
TARGET_SIZE = 1024

def preprocess_A(image):

    img = image.copy()

    # 1. Resize
    img_res = cv2.resize(
        img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA
    )

    # Split BGR
    b, g, r = cv2.split(img_res)

    # 2. Enhanced Green (Microaneurysms)
    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE
    )
    g_enhanced = clahe.apply(g)

    # 3. Raw Green (Structures / vessels)
    g_raw = g

    # 4. LAB Lightness (Exudates)
    lab = cv2.cvtColor(img_res, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)

    # 5. Channel-wise normalization
    g_enhanced = cv2.normalize(g_enhanced, None, 0, 255, cv2.NORM_MINMAX)
    g_raw = cv2.normalize(g_raw, None, 0, 255, cv2.NORM_MINMAX)
    l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)

    # uint8 type for YOLO Compatibility
    g_enhanced = g_enhanced.astype(np.uint8)
    g_raw = g_raw.astype(np.uint8)
    l = l.astype(np.uint8)

    # 6. Stack channels
    stacked_img = cv2.merge((g_enhanced, g_raw, l))

    return stacked_img
