"""
===========================================================
NeoLipi Segmentation v2
Tamil Character Segmentation
===========================================================
"""

import cv2
import numpy as np

from .common import (
    enhance_contrast,
    adaptive_binarize,
    strip_border,
    remove_small_noise,
    get_main_text_region,
    sort_boxes,
    crop_characters
)


# ==========================================================
# TAMIL PREPROCESSING
# ==========================================================

def preprocess_tamil(image):
    """
    Preprocess Tamil inscription image.
    Preserves rounded loops and smooth curves.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Contrast enhancement
    gray = enhance_contrast(gray)

    # Mild denoising
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold
    binary = adaptive_binarize(gray)

    # Preserve loops using closing
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # Remove tiny noise
    binary = remove_small_noise(binary, min_area=25)

    # Remove border
    binary = strip_border(binary)

    # Keep main text region
    binary = get_main_text_region(binary)

    return binary


# ==========================================================
# FIND CHARACTER BOXES
# ==========================================================

def find_character_boxes(binary):
    """
    Extract connected components
    likely to be Tamil characters.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    boxes = []

    H, W = binary.shape
    image_area = H * W

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 30:
            continue

        if area > image_area * 0.40:
            continue

        if w < 5 or h < 5:
            continue

        aspect = w / float(h)

        # Tamil characters are usually rounded
        if aspect > 3.5:
            continue

        if aspect < 0.20:
            continue

        boxes.append((x, y, w, h))

    return boxes


# ==========================================================
# REFINE BOXES
# ==========================================================

def refine_boxes(binary, boxes):
    """
    Tight bounding boxes around characters.
    """

    refined = []

    H, W = binary.shape

    for x, y, w, h in boxes:

        roi = binary[y:y+h, x:x+w]

        pts = cv2.findNonZero(roi)

        if pts is None:
            continue

        rx, ry, rw, rh = cv2.boundingRect(pts)

        refined.append((
            x + rx,
            y + ry,
            rw,
            rh
        ))

    return refined
# ==========================================================
# FILTER INVALID BOXES
# ==========================================================

def filter_boxes(boxes):
    """
    Remove invalid or noisy bounding boxes.
    """

    filtered = []

    for x, y, w, h in boxes:

        if w < 5 or h < 5:
            continue

        if w * h < 50:
            continue

        filtered.append((x, y, w, h))

    return filtered


# ==========================================================
# EXTRACT CHARACTERS
# ==========================================================

def extract_characters(binary, boxes):
    """
    Sort and crop Tamil characters.
    """

    boxes = sort_boxes(boxes)

    characters = crop_characters(
        binary,
        boxes,
        padding=3
    )

    return characters, boxes


# ==========================================================
# TAMIL SEGMENTATION
# ==========================================================

def segment_tamil(image, debug=False):
    """
    Complete Tamil segmentation pipeline.

    Parameters
    ----------
    image : ndarray
        Input image (BGR or grayscale)

    debug : bool
        Return debug image with bounding boxes

    Returns
    -------
    characters : list
    boxes : list
    binary : ndarray
    """

    # Step 1: Preprocess
    binary = preprocess_tamil(image)

    # Step 2: Connected Components
    boxes = find_character_boxes(binary)

    # Step 3: Refine Boxes
    boxes = refine_boxes(binary, boxes)

    # Step 4: Filter
    boxes = filter_boxes(boxes)

    # Step 5: Reading Order + Crop
    characters, boxes = extract_characters(
        binary,
        boxes
    )

    if debug:

        debug_img = cv2.cvtColor(
            binary,
            cv2.COLOR_GRAY2BGR
        )

        for x, y, w, h in boxes:

            cv2.rectangle(
                debug_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        return (
            characters,
            boxes,
            binary,
            debug_img
        )

    return (
        characters,
        boxes,
        binary
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    img = cv2.imread("test.png")

    chars, boxes, binary = segment_tamil(img)

    print(f"Characters detected : {len(chars)}")

    for i, ch in enumerate(chars):

        cv2.imwrite(
            f"tamil_char_{i}.png",
            ch
        )