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
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.abs(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    frame_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    frame_sharp = unsharp_mask(frame_eq, kernel_size=(7, 7), sigma=7.0, amount=1.2)
    return frame_sharp


def detect_ball_centroid(frame):
    h, w = frame.shape[:2]
    proc = preprocess(frame)

    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)

    # HSV red mask (blur-tolerant)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_hsv = cv2.inRange(hsv, lower_red1, upper_red1) | \
               cv2.inRange(hsv, lower_red2, upper_red2)

    # LAB red enhancement
    a_channel = lab[:, :, 1]
    mean_a = int(np.mean(a_channel))
    std_a = int(np.std(a_channel))
    thresh_a = max(150, mean_a + std_a // 2)
    _, mask_lab = cv2.threshold(a_channel, thresh_a, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(mask_hsv, mask_lab)

    # Morphology
    k = max(3, int(min(h, w) / 200))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = ww / float(hh) if hh > 0 else 0

        score = circularity * 2.0 + (area / (h * w)) * 10.0
        if aspect < 0.4 or aspect > 2.5:
            score *= 0.6

        if score > best_score and area > 50:
            best_score = score
            best_cnt = cnt

    if best_cnt is not None:
        return compute_centroid(best_cnt), mask

    # Fallback: Hough Circle
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) / 8,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=int(max(h, w) / 3)
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        cx, cy, r = max(circles, key=lambda c: c[2])
        return (cx, cy), mask

    return None, mask


if __name__ == "__main__":
    frame_files = sorted(os.listdir(FRAMES_DIR))
    trajectory = []

    OUTPUT_DIR = "tracking_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_path = os.path.join(OUTPUT_DIR, "tracking_video.mp4")
    writer = None

    for idx, file in enumerate(frame_files):
        frame = cv2.imread(os.path.join(FRAMES_DIR, file))
        if frame is None:
            continue

        centroid, mask = detect_ball_centroid(frame)

        if centroid is not None:
            trajectory.append((idx, centroid[0], centroid[1]))
            cv2.circle(frame, centroid, 6, (0, 0, 255), -1)
            cv2.putText(frame, f"{centroid}",
                        (centroid[0] + 10, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        vis = np.hstack([cv2.resize(frame, (mask_bgr.shape[1], mask_bgr.shape[0])), mask_bgr])

        if writer is None:
            h_vis, w_vis = vis.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, 25.0, (w_vis, h_vis))

        writer.write(vis)
        cv2.imshow("Ball Tracking (Frame | Mask)", vis)

        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    if writer:
        writer.release()

    cv2.destroyAllWindows()

    print("Trajectory length:", len(trajectory))
    print("First 10 points:", trajectory[:10])
