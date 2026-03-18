import cv2
import numpy as np

# Parameters tuned for maximum MA amplification and garbage reduction
CLIP_LIMIT_C = 2.5
TILE_SIZE_C = (4, 4)
STR_ELEM_SIZE = (7, 7) # Circular kernel size to match MA diameter
TARGET_SIZE = 1024 

def preprocess_C(image):
    if image is None:
        return None
        
    img = image.copy()

    # 1. Resize to target resolution
    img_res = cv2.resize(
        img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA
    )

    # Split BGR to isolate the Green channel
    b, g, r = cv2.split(img_res)

    # 2. Sharpening Step (Channel 1: Structural Detail)
    # High-pass filter to enhance fine edges and lesion boundaries.
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    g_sharp = cv2.filter2D(g, -1, kernel)
    
    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT_C, tileGridSize=TILE_SIZE_C
    )
    ch1_sharpened = clahe.apply(g_sharp)

    # 3. Black Top-Hat Transform (Channel 2: Lesion Isolation)
    # Target: Small dark spots (MAs/Hemorrhages). 
    # Logic: It subtracts the 'opened' image from the original, 
    # leaving behind only dark features smaller than the SE.
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, STR_ELEM_SIZE)
    ch2_tophat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)

    # 4. LAB Lightness (Channel 3: Global Context/Exudates)
    lab = cv2.cvtColor(img_res, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    ch3_lab = clahe.apply(l) # CLAHE applied for better exudate contrast

    # 5. Channel-wise normalization (Scaling to 0-255)
    ch1_sharpened = cv2.normalize(ch1_sharpened, None, 0, 255, cv2.NORM_MINMAX)
    ch2_tophat = cv2.normalize(ch2_tophat, None, 0, 255, cv2.NORM_MINMAX)
    ch3_lab = cv2.normalize(ch3_lab, None, 0, 255, cv2.NORM_MINMAX)

    # 6. Convert to uint8 
    # Essential for YOLOv8 processing and image saving
    ch1_sharpened = ch1_sharpened.astype(np.uint8)
    ch2_tophat = ch2_tophat.astype(np.uint8)
    ch3_lab = ch3_lab.astype(np.uint8)

    # 7. Stack channels: [Sharpened-Green, Black-TopHat, LAB-L]
    # This triplet provides YOLO with: 
    # [Edges/Detail, Specific-Pathology, Global-Context]
    stacked_img = cv2.merge((ch1_sharpened, ch2_tophat, ch3_lab))

    return stacked_img