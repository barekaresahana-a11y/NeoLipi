import cv2
import numpy as np

def standardize_stage4_crop(raw_crop, target_size=(128, 128)):
    if raw_crop is None or raw_crop.size == 0:
        return None

    # 1. Grayscale & Otsu Threshold
    gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY) if len(raw_crop.shape) == 3 else raw_crop.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Force black stroke on white background
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # 3. Tight crop around the main stroke contour
    inv_binary = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inv_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        glyph = binary[y:y+h, x:x+w]
    else:
        glyph = binary
        h, w = glyph.shape

    # 4. Add small square border padding (8% of glyph size)
    max_dim = max(h, w)
    pad = max(4, int(max_dim * 0.08))
    canvas_size = max_dim + (pad * 2)
    square_canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)

    off_x = (canvas_size - w) // 2
    off_y = (canvas_size - h) // 2
    square_canvas[off_y:off_y+h, off_x:off_x+w] = glyph

    # 5. Resize directly using INTER_AREA to keep sharp stroke features
    resized = cv2.resize(square_canvas, target_size, interpolation=cv2.INTER_AREA)
    _, sharp_binary = cv2.threshold(resized, 220, 255, cv2.THRESH_BINARY)

    return sharp_binary