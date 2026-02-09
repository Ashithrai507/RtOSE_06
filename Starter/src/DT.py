import cv2
import numpy as np
import os
import math
import csv

# ===================== USER CONFIG =====================
FRAMES_DIR = "frames_2sec"
OUTPUT_DIR = "tracking_output"

FPS = 240                  # <-- CHANGE THIS
PIXEL_TO_METER = 0.0023    # <-- CHANGE THIS (meters per pixel)

SMOOTHING_WINDOW = 5
# =======================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------ UTILITY FUNCTIONS ------------------

def compute_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return (
        int(M["m10"] / M["m00"]),
        int(M["m01"] / M["m00"])
    )


def unsharp_mask(image, kernel_size=(7, 7), sigma=7.0, amount=1.2):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    frame_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return unsharp_mask(frame_eq)


# ------------------ BALL DETECTION ------------------

def detect_ball_centroid(frame):
    h, w = frame.shape[:2]
    proc = preprocess(frame)

    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)

    # HSV red mask
    mask_hsv = (
        cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) |
        cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    )

    # LAB red enhancement
    a_channel = lab[:, :, 1]
    thresh_a = max(150, int(np.mean(a_channel) + np.std(a_channel) / 2))
    _, mask_lab = cv2.threshold(a_channel, thresh_a, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(mask_hsv, mask_lab)

    # Morphology
    k = max(3, int(min(h, w) / 200))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
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

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = ww / hh if hh > 0 else 0

        score = circularity * 2.0 + (area / (h * w)) * 10.0
        if aspect < 0.4 or aspect > 2.5:
            score *= 0.6

        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is not None:
        return compute_centroid(best_cnt), mask

    # Fallback: Hough Circle
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1.2, min(h, w) / 8,
        param1=50, param2=30,
        minRadius=5, maxRadius=int(max(h, w) / 3)
    )

    if circles is not None:
        cx, cy, r = max(np.round(circles[0]).astype(int), key=lambda c: c[2])
        return (cx, cy), mask

    return None, mask


# ------------------ PHYSICS ------------------

def compute_pixel_displacements(traj):
    return [
        math.hypot(traj[i+1][1] - traj[i][1],
                   traj[i+1][2] - traj[i][2])
        for i in range(len(traj) - 1)
    ]


def smooth(signal, window):
    if len(signal) < window:
        return signal
    return np.convolve(signal, np.ones(window) / window, mode="valid")


# ------------------ MAIN ------------------

frame_files = sorted(os.listdir(FRAMES_DIR))
trajectory = []

video_path = os.path.join(OUTPUT_DIR, "tracking_debug.mp4")
writer = None

for idx, fname in enumerate(frame_files):
    frame = cv2.imread(os.path.join(FRAMES_DIR, fname))
    if frame is None:
        continue

    centroid, mask = detect_ball_centroid(frame)

    if centroid:
        trajectory.append((idx, centroid[0], centroid[1]))
        cv2.circle(frame, centroid, 6, (0, 0, 255), -1)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    vis = np.hstack([frame, mask_bgr])

    if writer is None:
        h, w = vis.shape[:2]
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (w, h)
        )

    writer.write(vis)
    cv2.imshow("Tracking (Frame | Mask)", vis)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

if writer:
    writer.release()
cv2.destroyAllWindows()

print(f"[INFO] Tracked frames: {len(trajectory)}")

# ------------------ DISTANCE & SPEED ------------------

pixel_dist = compute_pixel_displacements(trajectory)
dist_m = [d * PIXEL_TO_METER for d in pixel_dist]

speed_mps = [d * FPS for d in dist_m]
speed_kmph = [v * 3.6 for v in speed_mps]
speed_smooth = smooth(speed_kmph, SMOOTHING_WINDOW)

print(f"Peak Speed    : {max(speed_smooth):.2f} km/h")
print(f"Average Speed : {sum(speed_smooth)/len(speed_smooth):.2f} km/h")

# ------------------ CSV EXPORT ------------------

csv_path = os.path.join(OUTPUT_DIR, "speed_output.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "x", "y", "speed_kmph"])
    for i, v in enumerate(speed_kmph):
        frame, x, y = trajectory[i + 1]
        writer.writerow([frame, x, y, v])

print(f"[INFO] CSV saved to {csv_path}")
