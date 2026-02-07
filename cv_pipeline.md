# Classic Computer Vision Pipeline for Robust Red-Ball Detection & Tracking

Overview
- Purpose: reliable detection and tracking of a single red ball, robust to blur, low contrast, and spurious red objects.
- Scope: single-camera frames or video, classical CV (OpenCV), no deep learning required.
- Outputs: per-frame centroid (x,y), optional mask, visualization video, CSV trajectory.

Table of contents
- Acquisition
- Preprocessing
- Segmentation (color & edge)
- Morphology and cleanup
- Candidate extraction & filtering
- Fallbacks (Hough circles / edge-based)
- Tracking & temporal filtering
- Post-processing & outputs
- Tuning for blurred frames
- Metrics & evaluation
- Troubleshooting / tips
- Minimal Python skeleton

**1. Acquisition**
- Capture source: video file or camera (cv2.VideoCapture).
- Normalize: resize frames to a fixed working size (e.g., 640×360 or 1280×720) for deterministic kernels and performance.
- Maintain consistent color space: OpenCV reads BGR.

**2. Preprocessing**
Goal: boost contrast and recover edges/detail degraded by blur.
Recommended steps:
- Resize: cv2.resize(frame, target).
- Denoise (optional): cv2.medianBlur(frame, 3–7) or cv2.bilateralFilter for texture preservation.
- CLAHE: apply to LAB L-channel
  - clipLimit: 2.0 (range 1.0–4.0)
  - tileGridSize: (8,8)
- Sharpen (unsharp mask):
  - blur kernel: (5–9,5–9), sigma 5–10
  - amount: 1.0–1.6
- Convert to color spaces for segmentation:
  - HSV (for hue/sat)
  - LAB (for 'a' channel — red-green axis)
Notes:
- For heavy blur, increase CLAHE clipLimit and unsharp amount carefully to avoid artifacts.

**3. Segmentation**
A. Color-based segmentation (primary):
- HSV red ranges (two ranges due to hue wrap-around):
  - lower1 = [0, Sat_min, Val_min]
  - upper1 = [10, 255, 255]
  - lower2 = [170, Sat_min, Val_min]
  - upper2 = [180, 255, 255]
- Suggested Sat_min: 50–80 (lower for faded/blurred balls)
- Suggested Val_min: 40–80 (depending on illumination)

B. LAB segmentation (complementary):
- Use 'a' channel; red pixels often have higher 'a'.
- Adaptive threshold: thresh = max(140–160, mean(a) + k*std(a)), with k ≈ 0.3–0.7.
- Binary mask from threshold.

C. Combine masks:
- mask = mask_hsv OR mask_lab
- Optionally combine with edge mask (Canny) dilated to allow for blurred edges.

**4. Morphology & cleanup**
- Kernel size proportional to image: k = max(3, int(min(h,w)/200)); make odd.
- Use MORPH_OPEN (iterations 1) to remove small specks.
- Use MORPH_CLOSE (iterations 1–3) to fill holes and connect fragments.
- medianBlur on mask (ksize 3–7) to reduce salt-and-pepper.
- Optional hole filling: floodFill or contour filling for internal holes.

**5. Candidate extraction & filtering**
- contours = cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
- For each contour compute:
  - area = cv2.contourArea(cnt)
  - perimeter = cv2.arcLength(cnt, True)
  - circularity = 4π * area / (perimeter^2)  (range 0–1; 1=perfect circle)
  - bounding box aspect = w/h
  - solidity = area / convexHullArea
  - color homogeneity inside contour: stddev of LAB channels (low stddev preferred)
Filters (suggested):
- area > min_area (e.g., 50–200 px depending on resize)
- circularity > 0.5–0.7 (increase to reject non-circular)
- solidity > 0.8
- aspect between 0.6–1.6
- color homogeneity: std(L,A,B) small relative to object mean
Scoring:
- score = α * circularity + β * normalized_area + γ * solidity + δ * color_consistency
- Choose best-scoring contour above a threshold.

**6. Fallbacks**
- HoughCircles on blurred grayscale (good for blurred circular objects):
  - gray = cv2.medianBlur(gray, 5–7)
  - dp = 1.0–1.5
  - minDist = min(h,w)/8
  - param1 (Canny threshold) ≈ 50–100
  - param2 (accumulator threshold) ≈ 20–40 (lower for more sensitivity)
  - minRadius ≈ 5–10 px, maxRadius ≈ max(h,w)/3
- Use Hough result only if color/morphological masks failed OR if the circle has color agreement with red.

**7. Tracking & temporal filtering**
- Use lightweight filters to stabilize centroid:
  - Kalman filter (constant velocity) — OpenCV Kalman or simple custom.
  - Exponential moving average (EMA) on (x,y): x_smoothed = α x_new + (1-α) x_prev (α 0.2–0.6).
- Motion gating:
  - Reject detections that imply impossible instantaneous speed (use prior frame dt and expected max speed).
- Temporal continuity:
  - Prefer detections that persist for N consecutive frames (N=1–3) or give them higher confidence.

**8. Post-processing & outputs**
- Draw visualization (centroid, bounding circle, trajectory).
- Save:
  - Visualization video: OpenCV VideoWriter, fourcc "mp4v" or "XVID".
  - CSV: frame_index, x, y, radius, score.
  - Optional per-frame mask PNGs for debugging.
- Logging: save per-frame candidate stats (area, circularity, solidity, color_std).

**9. Tuning for blurred single frames**
- Lower HSV saturation threshold and rely more on LAB 'a' channel.
- Increase morphological closing to join fragmented patches.
- Use unsharp mask with moderate amount to recover edges.
- Use HoughCircles as a trusted fallback; combine with color check inside the circle.
- Increase allowed radius variance and lower min area thresholds when object is small/blurred.
- Add color homogeneity acceptance, as blur tends to smooth internal color — accept slightly higher stddev.

**10. Metrics & evaluation**
- If ground-truth available:
  - Centroid error: mean Euclidean distance, median, max.
  - Detection rate: true positives / (true positives + false negatives).
  - False positives per frame.
  - Mask IoU (if masks labeled).
- Logging suggested: save detection outcomes and scores for offline analysis.

**11. Troubleshooting / tips**
- Visualize side-by-side: frame | mask | edges to identify failure modes.
- Save failure frames and inspect HSV/LAB distributions.
- Tune thresholds on representative frames; use a small GUI or notebook to slide parameters.
- If many false positives from other red objects:
  - Increase circularity/solidity thresholds.
  - Add size constraints (min/max radius).
  - Require temporal persistence or motion consistency.
- If detection disappears intermittently:
  - Soften thresholds and rely on tracker to interpolate short gaps.

**12. Minimal Python skeleton**
- The repository already contains a full implementation. Sketch of loop:
```python
cap = cv2.VideoCapture("input.mp4")
trajectory = []
while True:
    ret, frame = cap.read()
    if not ret: break
    proc = preprocess(frame)                # CLAHE + unsharp
    mask = color_segment(proc)              # HSV + LAB
    mask = clean_mask(mask, frame.shape)    # morphology
    centroid = select_ball_contour(mask)    # contour filters
    if centroid is None:
        centroid = hough_fallback(proc)
    centroid = temporal_filter(centroid)
    visualize_and_save(frame, mask, centroid)
cap.release()
```

**13. Recommended default parameters (starting point)**
- Resize: width = 640
- CLAHE: clipLimit=2.0, tileGridSize=(8,8)
- Unsharp: kernel=(7,7), sigma=7.0, amount=1.2
- HSV sat_min = 60, val_min = 50
- LAB a threshold: max(150, mean + std//2)
- Morph kernel k = max(3, min(h,w)//200), median blur k=5
- Circularity threshold: 0.55
- Solidity threshold: 0.8
- Hough: dp=1.2, param1=50, param2=30, minRadius=5

**14. Final notes**
- Use a small labeled dataset to tune thresholds and evaluate metrics.
- Add a CLI flag to save masks/videos and to adjust parameters at runtime.
- For highly challenging scenarios (heavy motion blur, occlusions, complex backgrounds), consider switching to a lightweight learned detector (e.g., small YOLO) or combining optical-flow based tracking.

References
- OpenCV documentation: color conversions, findContours, HoughCircles, VideoWriter.
- Classic circle detection and morphological techniques.

End of document.