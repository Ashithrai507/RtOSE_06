import cv2
import numpy as np
import os

FRAMES_DIR = "frames_2sec"


def compute_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def unsharp_mask(image, kernel_size=(9, 9), sigma=10.0, amount=1.5, threshold=0):
    """Return a sharpened version of the image using an unsharp mask.

    Helps recover edges in slightly blurred images.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.abs(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess(frame):
    """Apply contrast-limited adaptive histogram equalization and sharpening."""
    # CLAHE on L channel (for better contrast under blur)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    frame_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Unsharp mask to enhance edges in blurred frames
    frame_sharp = unsharp_mask(frame_eq, kernel_size=(7, 7), sigma=7.0, amount=1.2)
    return frame_sharp


def detect_ball_centroid(frame):
    """Detect ball centroid robustly in possibly blurred frames.

    Returns (cx, cy) or None. Uses combined HSV+LAB color masks, morphological cleanup,
    circularity filter, and HoughCircles fallback.
    """
    h, w = frame.shape[:2]
    # Preprocess frame to improve contrast and edges
    proc = preprocess(frame)

    # Convert to HSV and LAB for robust color detection
    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)

    # HSV red ranges with lower saturation threshold to tolerate blur/desaturation
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # LAB: red tends to have higher 'a' channel values; threshold adaptively
    a_channel = lab[:, :, 1]
    # adaptive threshold: mean + k*std
    mean_a = int(np.mean(a_channel))
    std_a = int(np.std(a_channel))
    thresh_a = max(150, mean_a + std_a // 2)
    _, mask_lab = cv2.threshold(a_channel, thresh_a, 255, cv2.THRESH_BINARY)

    # Combine masks to be more robust
    mask = cv2.bitwise_or(mask_hsv, mask_lab)

    # Morphological cleanup: kernel scaled to image size
    k = max(3, int(min(h, w) / 200))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)

    # Find contours and pick the most circular/likely one
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # compute bounding box to reject extremely elongated shapes
        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = ww / float(hh) if hh > 0 else 0
        candidates.append((cnt, area, circularity, aspect))

    # Prefer circular contours with reasonable area and aspect ratio
    best = None
    best_score = 0
    for cnt, area, circularity, aspect in candidates:
        # score balances area and circularity; tweak weights as needed
        score = circularity * 2.0 + (area / (h * w)) * 10.0
        # penalize extreme aspect ratios
        if aspect < 0.4 or aspect > 2.5:
            score *= 0.6
        if score > best_score and area > 50:
            best_score = score
            best = cnt

    if best is not None:
        return compute_centroid(best), mask

    # Fallback: try Hough Circle detection on grayscale (works for blur if circle-like)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    # dp=1.2, minDist ~ min(h,w)/8
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) / 8,
                               param1=50, param2=30, minRadius=5, maxRadius=int(max(h, w) / 3))
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # pick the largest circle (by radius)
        circles = sorted(circles, key=lambda c: c[2], reverse=True)
        cx, cy, r = circles[0]
        return (int(cx), int(cy)), mask

    return None, mask


if __name__ == "__main__":
    frame_files = sorted(os.listdir(FRAMES_DIR))
    trajectory = []
    # create output directory for tracking video
    OUTPUT_DIR = "tracking_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _video_writer = None
    _video_path = os.path.join(OUTPUT_DIR, "tracking_video.mp4")

    for idx, file in enumerate(frame_files):
        path = os.path.join(FRAMES_DIR, file)
        frame = cv2.imread(path)
        if frame is None:
            continue

        centroid, mask = detect_ball_centroid(frame)
        if centroid:
            trajectory.append((idx, centroid[0], centroid[1]))
            cv2.circle(frame, centroid, 6, (0, 0, 255), -1)
            cv2.putText(frame, f"{centroid}", (centroid[0] + 10, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # show mask + frame side by side for debugging
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        vis = np.hstack([cv2.resize(frame, (mask_bgr.shape[1], mask_bgr.shape[0])), mask_bgr])
        # initialize video writer lazily with vis size
        if _video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h_vis, w_vis = vis.shape[:2]
            # VideoWriter expects (width, height)
            _video_writer = cv2.VideoWriter(_video_path, fourcc, 25.0, (w_vis, h_vis))
        # write the visualization frame to output video (convert to BGR if needed)
        try:
            _video_writer.write(vis)
        except Exception:
            # if writer fails (e.g., wrong type), convert vis to BGR
            _video_writer.write(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        cv2.imshow("Ball Centroid Tracking (frame | mask)", vis)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    # release video writer if created
    if _video_writer is not None:
        _video_writer.release()
    cv2.destroyAllWindows()
