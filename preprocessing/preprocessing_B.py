import cv2
import numpy as np

# ----------------------- Pipeline B: Sharp-Green Stacking ----------------------- #

# Parameters specifically tuned for detecting smaller lesions (MAs)
CLIP_LIMIT_B = 2.5
TILE_SIZE_B = (4, 4)  
TARGET_SIZE = 1024 

def preprocess_B(image):
    
    if image is None:
        return None
        
    img = image.copy()

    # 1. Resize to target resolution
    img_res = cv2.resize(
        img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA
    )

    # Split BGR to isolate the Green channel (highest contrast for retinal vessels/MAs)
    b, g, r = cv2.split(img_res)

    # 2. Sharpening Step
    # This kernel enhances the center pixel relative to its neighbors to make MAs pop against the background.
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    g_sharp = cv2.filter2D(g, -1, kernel)

    # 3. Enhanced Sharpened Green (Channel 1)
    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT_B, tileGridSize=TILE_SIZE_B
    )
    g_enhanced = clahe.apply(g_sharp)

    # 4. Raw Green (Channel 2: For general anatomy/vessels)
    g_raw = g

    # 5. LAB Lightness (Channel 3: For Exudates/Overall Structure)
    lab = cv2.cvtColor(img_res, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)

    # 6. Channel-wise normalization (Scaling to 0-255)
    g_enhanced = cv2.normalize(g_enhanced, None, 0, 255, cv2.NORM_MINMAX)
    g_raw = cv2.normalize(g_raw, None, 0, 255, cv2.NORM_MINMAX)
    l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)

    # 7. Convert to uint8 
    g_enhanced = g_enhanced.astype(np.uint8)
    g_raw = g_raw.astype(np.uint8)
    l = l.astype(np.uint8)

    # 8. Stack channels: [Enhanced-Sharp-Green, Raw-Green, LAB-L]
    stacked_img = cv2.merge((g_enhanced, g_raw, l))

    return stacked_img