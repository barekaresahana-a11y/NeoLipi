import cv2
import numpy as np

def strip_border(binary_image, ratio=0.015, min_margin=4):
    """
    Blank out a thin band around the outer edge of the page.
    Stone/manuscript photos almost always have a rough physical edge
    or scan border that adaptive thresholding turns into a tall thin
    strip of "foreground" — that strip is not text, but it's regular
    and thin enough to pass the normal character-size filters. Zeroing
    it out before any component analysis removes it at the source.
    """
    h, w = binary_image.shape
    my = max(min_margin, int(h * ratio))
    mx = max(min_margin, int(w * ratio))

    cleaned = binary_image.copy()
    cleaned[:my, :] = 0
    cleaned[-my:, :] = 0
    cleaned[:, :mx] = 0
    cleaned[:, -mx:] = 0
    return cleaned


def remove_small_noise(binary_image, min_area=15):
    """
    Drop connected components below min_area. Run this BEFORE line
    detection / shirorekha removal — scattered speckle noise (grain,
    stains, worm holes) otherwise pollutes the row projection profile
    that get_text_lines() and remove_shirorekha() rely on.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    clean = np.zeros_like(binary_image)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def get_main_text_region(binary_image, pad=15):
    """
    Isolate the single largest, ink-densest block of text on the page
    and zero out everything else (side annotations, folio numbers,
    decorative borders, background texture).

    Strategy: dilate heavily so that characters within a line/paragraph
    fuse into one solid blob, while the gap to marginal notes or a
    decorative border (which are spatially separated from the main
    block) is preserved. Then pick the resulting connected component
    with the most actual ink pixels underneath it (not just the
    largest bounding box — a long thin decorative border can have a
    large box but very little ink compared to a dense paragraph).
    """
    h, w = binary_image.shape

    # Scale the dilation kernel to the text size on this page so the
    # same function works across different image resolutions.
    lines = get_text_lines(binary_image)
    if lines:
        median_line_h = int(np.median([y2 - y1 for y1, y2 in lines]))
    else:
        median_line_h = 25
    kernel_size = max(15, median_line_h)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary_image, kernel, iterations=3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )

    if num_labels <= 1:
        return binary_image.copy()

    best_label, best_ink = None, -1
    for i in range(1, num_labels):
        ink = np.count_nonzero(binary_image[labels == i])
        if ink > best_ink:
            best_ink = ink
            best_label = i

    x = stats[best_label, cv2.CC_STAT_LEFT]
    y = stats[best_label, cv2.CC_STAT_TOP]
    bw = stats[best_label, cv2.CC_STAT_WIDTH]
    bh = stats[best_label, cv2.CC_STAT_HEIGHT]

    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)

    mask = np.zeros_like(binary_image)
    mask[y0:y1, x0:x1] = binary_image[y0:y1, x0:x1]
    return mask


def get_text_lines(binary_image, min_gap=3, min_height=10):
    """
    Split a full-page binary image into text-line bands using the
    horizontal (row-wise) projection profile. Returns a list of
    (y_start, y_end) tuples, one per detected line.
    """
    row_sums = np.sum(binary_image > 0, axis=1)
    rows_with_text = row_sums > 0

    lines = []
    start = None
    gap = 0

    for i, has_text in enumerate(rows_with_text):
        if has_text:
            if start is None:
                start = i
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > min_gap:
                    end = i - gap
                    if end - start >= min_height:
                        lines.append((start, end))
                    start = None
                    gap = 0

    if start is not None:
        end = len(rows_with_text) - 1
        if end - start >= min_height:
            lines.append((start, end))

    return lines


def remove_shirorekha(binary_image):
    """
    Remove the Devanagari headline (Shirorekha) while preserving
    character bodies.

    Instead of one global horizontal-open pass (which misses lines
    whose headline sits at a different height, and which used to be
    undone by a later MORPH_CLOSE), this detects each text line first,
    then locates and blanks out the dense horizontal headline band
    *within that line only*. No re-closing is done afterwards, so the
    gap this creates between characters is preserved.
    """

    cleaned = binary_image.copy()
    lines = get_text_lines(cleaned)

    # Fallback: if line detection fails (e.g. a single-line crop),
    # treat the whole image as one line.
    if not lines:
        lines = [(0, cleaned.shape[0] - 1)]

    for (y1, y2) in lines:
        line = cleaned[y1:y2 + 1, :]

        row_sums = np.sum(line > 0, axis=1)
        if row_sums.max() == 0:
            continue

        # The headline is the row with the most foreground pixels in
        # this line. Expand up/down while the row stays "dense" to
        # capture its full thickness (headlines are usually 2-4px).
        peak = int(np.argmax(row_sums))
        threshold = row_sums[peak] * 0.5

        top = peak
        while top > 0 and row_sums[top - 1] > threshold:
            top -= 1

        bottom = peak
        while bottom < len(row_sums) - 1 and row_sums[bottom + 1] > threshold:
            bottom += 1

        line[top:bottom + 1, :] = 0
        cleaned[y1:y2 + 1, :] = line

    return cleaned

def split_wide_component(x, y, w, h, binary):

    roi = binary[y:y+h, x:x+w]

    projection = np.sum(roi > 0, axis=0)

    if len(projection) < 10:
        return [(x, y, w, h)]

    # Smooth the projection to reduce noise
    projection = cv2.GaussianBlur(
        projection.astype(np.float32).reshape(1, -1),
        (1, 5),
        0
    ).flatten()

    threshold = projection.max() * 0.35

    split_points = []

    # Find local minima
    for i in range(2, len(projection) - 2):

        if (projection[i] < projection[i - 1] and
            projection[i] < projection[i + 1] and
            projection[i] < threshold):
            split_points.append(i)

    if not split_points:
        return [(x, y, w, h)]

    boxes = []
    start = 0

    for p in split_points:

        if p - start >= 10:
            boxes.append((x + start, y, p - start, h))
            start = p

    if w - start >= 10:
        boxes.append((x + start, y, w - start, h))

    return boxes

def segment_devanagari(binary_image):

    work = binary_image.copy()

    # 1. Strip the physical page/stone edge so it can't be mistaken for text.
    work = strip_border(work)

    # 2. Remove speckle noise so it doesn't corrupt the row-projection
    #    profile used for line detection and headline removal.
    work = remove_small_noise(work, min_area=15)

    # 3. Keep only the main, densest text block — drops side annotations,
    #    folio/page numbers, and decorative borders.
    work = get_main_text_region(work)

    # 4. Remove Shirorekha (per-line, adaptive band removal).
    #    NOTE: deliberately no MORPH_CLOSE afterwards — closing here would
    #    re-bridge the very gaps shirorekha removal just created, which is
    #    what was causing whole words to stay as one connected component.
    work = remove_shirorekha(work)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    work = cv2.morphologyEx(
        work,
        cv2.MORPH_OPEN,
        kernel
    )

    return work


def segment_characters(binary_image, script):
    """
    Neat, character-by-character segmentation for Devanagari, Tamil, and Brahmi.
    """
    work_binary = binary_image.copy()

    if script == "Devanagari":
        work_binary = segment_devanagari(binary_image)

        # ------------------------------------------
        # Remove very small connected components
        # ------------------------------------------

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            work_binary,
            connectivity=8
        )

        clean = np.zeros_like(work_binary)

        for i in range(1, num_labels):

            area = stats[i, cv2.CC_STAT_AREA]

            # Ignore tiny blobs
            if script == "Devanagari":
                if area >= 40:
                    clean[labels == i] = 255
            else:
                if area >= 70:
                    clean[labels == i] = 255

        work_binary = clean

    # ------------------------------------------
    # Find contours
    # ------------------------------------------

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        work_binary,
        connectivity=8
    )

    raw_boxes = []

    img_h, img_w = work_binary.shape

    # Character size and aspect ratio thresholds.
    # These are defined locally here since they are needed for bounding
    # box filtering in the connected component loop.
    if script == "Devanagari":
        min_area = 40
        max_area = 5000
        min_w, max_w = 4, 80
        min_h, max_h = 10, 120
        min_aspect, max_aspect = 0.2, 5.0
    else:
        min_area = 70
        max_area = 7500
        min_w, max_w = 5, 90
        min_h, max_h = 12, 140
        min_aspect, max_aspect = 0.25, 4.5

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        if area > max_area:
            continue

        if w < min_w or w > max_w:
            continue

        if h < min_h or h > max_h:
            continue

        aspect = w / float(h)

        if aspect < min_aspect or aspect > max_aspect:
            continue

        if x <= 2 or y <= 2:
            continue

        if x + w >= img_w - 2:
            continue

        if y + h >= img_h - 2:
            continue

        if script == "Devanagari" and w > 32:
            raw_boxes.extend(
                split_wide_component(
                    x,
                    y,
                    w,
                    h,
                    work_binary
                )
            )
        else:
            raw_boxes.append((x, y, w, h))

    # 3. Sort Bounding Boxes Reading Order (Top-to-Bottom, Left-to-Right)
    # Group boxes into horizontal lines first, then sort left-to-right
    if not raw_boxes:
        return [], []

    raw_boxes = sorted(raw_boxes, key=lambda b: b[1]) # Sort by Y

    sorted_boxes = []
    line_threshold = 20  # Max vertical offset to consider on same line

    current_line = [raw_boxes[0]]
    for box in raw_boxes[1:]:
        if abs(box[1] - current_line[0][1]) < line_threshold:
            current_line.append(box)
        else:
            # Sort current line left to right
            current_line.sort(key=lambda b: b[0])
            sorted_boxes.extend(current_line)
            current_line = [box]

    if current_line:
        current_line.sort(key=lambda b: b[0])
        sorted_boxes.extend(current_line)

    # 4. Crop Character Images from ORIGINAL binary (not shirorekha-removed)
    characters = []
    for (x, y, w, h) in sorted_boxes:
        roi = binary_image[y:y+h, x:x+w]

        # Invert roi if needed to ensure character is white on black background
        if np.mean(roi) > 127:
            roi = cv2.bitwise_not(roi)

        roi = cv2.resize(roi, (64, 64))
        characters.append(roi)

    return characters, sorted_boxes