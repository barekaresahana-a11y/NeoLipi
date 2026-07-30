"""
===========================================================
NeoLipi Segmentation v2
Devanagari Character Segmentation
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
    merge_components,
    split_projection,
    sort_boxes,
    crop_characters
)


# ==========================================================
# PREPROCESSING
# ==========================================================

def preprocess_devanagari(image):
    """
    Preprocess Devanagari inscription image.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Improve contrast
    gray = enhance_contrast(gray)

    # Remove stone texture while preserving edges
    gray = cv2.bilateralFilter(
        gray,
        7,
        60,
        60
    )

    # Adaptive threshold
    binary = adaptive_binarize(gray)

    # Remove tiny blobs
    binary = remove_small_noise(binary, min_area=25)

    # Remove border
    binary = strip_border(binary)

    # Keep only inscription
    binary = get_main_text_region(binary)

    return binary


# ==========================================================
# SHIROREKHA REMOVAL
# ==========================================================

def remove_shirorekha(binary):
    """
    Remove the horizontal headline
    connecting Devanagari characters.
    """

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (35, 1)
    )

    detected = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    cleaned = cv2.subtract(
        binary,
        detected
    )

    return cleaned


# ==========================================================
# FIND CHARACTER BOXES
# ==========================================================

def find_character_boxes(binary):
    """
    Connected component analysis.
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

        if area < 25:
            continue

        if area > image_area * 0.40:
            continue

        if w < 4 or h < 6:
            continue

        aspect = w / float(h)

        if aspect > 4.0:
            continue

        if aspect < 0.15:
            continue

        boxes.append((x, y, w, h))

    return boxes


# ==========================================================
# MERGE SMALL FRAGMENTS
# ==========================================================

def merge_devanagari_fragments(boxes):
    """
    Merge broken character fragments.
    """

    return merge_components(
        boxes,
        x_gap=8,
        y_overlap_ratio=0.60,
        height_ratio=0.50
    )


# ==========================================================
# SPLIT JOINED CHARACTERS
# ==========================================================

def split_joined_boxes(binary, boxes):
    """
    Split wide connected components.
    """

    final_boxes = []

    for x, y, w, h in boxes:

        roi = binary[y:y+h, x:x+w]

        parts = split_projection(
            roi,
            min_width_ratio=1.6
        )

        if len(parts) == 1:

            final_boxes.append((x, y, w, h))

        else:

            offset = x

            for part in parts:

                ph, pw = part.shape

                final_boxes.append(
                    (
                        offset,
                        y,
                        pw,
                        ph
                    )
                )

                offset += pw

    return final_boxes
# ==========================================================
# REFINE BOXES
# ==========================================================

def refine_boxes(binary, boxes):
    """
    Tighten bounding boxes around foreground pixels.
    """

    refined = []

    H, W = binary.shape

    for x, y, w, h in boxes:

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(W, x + w)
        y2 = min(H, y + h)

        roi = binary[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        pts = cv2.findNonZero(roi)

        if pts is None:
            continue

        rx, ry, rw, rh = cv2.boundingRect(pts)

        refined.append((
            x1 + rx,
            y1 + ry,
            rw,
            rh
        ))

    return refined


# ==========================================================
# FILTER INVALID BOXES
# ==========================================================

def filter_boxes(boxes):
    """
    Remove noisy or invalid boxes.
    """

    filtered = []

    for x, y, w, h in boxes:

        if w < 4 or h < 6:
            continue

        if w * h < 40:
            continue

        filtered.append((x, y, w, h))

    return filtered


# ==========================================================
# EXTRACT CHARACTERS
# ==========================================================

def extract_characters(binary, boxes):
    """
    Sort boxes and crop character images.
    """

    boxes = sort_boxes(boxes)

    characters = crop_characters(
        binary,
        boxes,
        padding=3
    )

    return characters, boxes


# ==========================================================
# DEVANAGARI SEGMENTATION
# ==========================================================

def segment_devanagari(image, debug=False):
    """
    Complete Devanagari segmentation pipeline.

    Returns
    -------
    characters
    boxes
    binary
    """

    # Step 1 : Preprocess
    binary = preprocess_devanagari(image)

    # Step 2 : Remove Shirorekha
    binary = remove_shirorekha(binary)

    # Step 3 : Connected Components
    boxes = find_character_boxes(binary)

    # Step 4 : Merge Broken Parts
    boxes = merge_devanagari_fragments(boxes)

    # Step 5 : Split Wide Characters
    boxes = split_joined_boxes(binary, boxes)

    # Step 6 : Refine Boxes
    boxes = refine_boxes(binary, boxes)

    # Step 7 : Remove Noise Boxes
    boxes = filter_boxes(boxes)

    # Step 8 : Extract Characters
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

    chars, boxes, binary = segment_devanagari(img)

    print(f"Characters detected : {len(chars)}")

    for i, ch in enumerate(chars):

        cv2.imwrite(
            f"devanagari_char_{i}.png",
            ch
        )