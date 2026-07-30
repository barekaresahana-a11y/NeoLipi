"""
===========================================================
NeoLipi Segmentation V2
Character Segmentation
===========================================================
"""

import cv2
import numpy as np


# ==========================================================
# FIND CHARACTER COMPONENTS
# ==========================================================

def find_character_boxes(
    binary,
    min_area=50,
    min_width=5,
    min_height=8
):
    """
    Detect connected components corresponding to characters.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    height, width = binary.shape

    boxes = []

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        if w < min_width:
            continue

        if h < min_height:
            continue

        if w > width * 0.95 and h > height * 0.95:
            continue

        boxes.append((x, y, w, h))

    return boxes


# ==========================================================
# MERGE BROKEN COMPONENTS
# ==========================================================

def merge_boxes(
    boxes,
    x_gap=5,
    y_overlap_ratio=0.5
):

    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])

    merged = []

    current = list(boxes[0])

    for box in boxes[1:]:

        x, y, w, h = box

        cx, cy, cw, ch = current

        current_right = cx + cw
        box_right = x + w

        gap = x - current_right

        overlap = max(
            0,
            min(cy + ch, y + h) - max(cy, y)
        )

        overlap_ratio = overlap / max(
            1,
            min(ch, h)
        )

        if gap <= x_gap and overlap_ratio >= y_overlap_ratio:

            nx = min(cx, x)
            ny = min(cy, y)

            nr = max(current_right, box_right)
            nb = max(cy + ch, y + h)

            current = [
                nx,
                ny,
                nr - nx,
                nb - ny
            ]

        else:

            merged.append(tuple(current))

            current = [x, y, w, h]

    merged.append(tuple(current))

    return merged


# ==========================================================
# REMOVE NESTED BOXES
# ==========================================================

def remove_nested_boxes(boxes):

    filtered = []

    for i, b1 in enumerate(boxes):

        keep = True

        x1, y1, w1, h1 = b1

        for j, b2 in enumerate(boxes):

            if i == j:
                continue

            x2, y2, w2, h2 = b2

            if (
                x1 >= x2 and
                y1 >= y2 and
                x1 + w1 <= x2 + w2 and
                y1 + h1 <= y2 + h2
            ):
                keep = False
                break

        if keep:
            filtered.append(b1)

    return filtered


# ==========================================================
# SORT BOXES
# ==========================================================

def sort_boxes(boxes):

    return sorted(boxes, key=lambda b: b[0])


# ==========================================================
# ADD PADDING
# ==========================================================

def add_padding(
    boxes,
    image_shape,
    padding=3
):

    h, w = image_shape

    padded = []

    for x, y, bw, bh in boxes:

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)

        padded.append(
            (
                x1,
                y1,
                x2 - x1,
                y2 - y1
            )
        )

    return padded


# ==========================================================
# REFINE BOXES
# ==========================================================

def refine_boxes(
    binary,
    boxes
):

    boxes = merge_boxes(boxes)

    boxes = remove_nested_boxes(boxes)

    boxes = sort_boxes(boxes)

    boxes = add_padding(
        boxes,
        binary.shape
    )

    return boxes


# ==========================================================
# RESIZE CHARACTER
# ==========================================================

def resize_character(
    image,
    size=64
):

    h, w = image.shape

    if h == 0 or w == 0:

        return np.zeros(
            (size, size),
            dtype=np.uint8
        )

    scale = min(
        (size - 8) / w,
        (size - 8) / h
    )

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    x = (size - new_w) // 2
    y = (size - new_h) // 2

    canvas[
        y:y+new_h,
        x:x+new_w
    ] = resized

    return canvas


# ==========================================================
# EXTRACT CHARACTERS
# ==========================================================

def extract_characters(
    binary,
    boxes,
    output_size=64
):

    characters = []

    valid_boxes = []

    for x, y, w, h in boxes:

        padding = 5

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(binary.shape[1], x + w + padding)
        y2 = min(binary.shape[0], y + h + padding)

        char = binary[y1:y2, x1:x2]

        if char.size == 0:
            continue

        char = resize_character(
            char,
            output_size
        )

        characters.append(char)

        valid_boxes.append(
            (x, y, w, h)
        )

    return characters, valid_boxes


# ==========================================================
# MAIN CHARACTER SEGMENTATION
# ==========================================================

def segment_characters(line_image):
    """
    Segment one text line into individual characters.
    """

    boxes = find_character_boxes(line_image)

    boxes = refine_boxes(
        line_image,
        boxes
    )

    characters, boxes = extract_characters(
        line_image,
        boxes
    )

    return characters, boxes


# ==========================================================
# DEBUG
# ==========================================================

if __name__ == "__main__":

    img = cv2.imread(
        "line.png",
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        print("line.png not found")

        exit()

    chars, boxes = segment_characters(img)

    print("Detected:", len(chars))

    debug = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2BGR
    )

    for x, y, w, h in boxes:

        cv2.rectangle(
            debug,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Characters", debug)

    cv2.waitKey(0)

    cv2.destroyAllWindows()