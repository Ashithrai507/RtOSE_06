## Phase 1 RtOSE
```json
Pre-recorded Video
        ↓
Frame Extraction
        ↓
Object Segmentation (per frame)
        ↓
Centroid Extraction
        ↓
Pixel Displacement Calculation
        ↓
Reference-Based Pixel → Meter Scaling
        ↓
Distance Validation
        ↓
Speed Computation
```

### Step 1: Video Input & Frame Extraction
Input: Recorded mobile video
- Known FPS (or extracted from metadata)
- Process video frame-by-frame
- No frame skipping in Phase 1

## Phase 1 — RtOSE

This document describes Phase 1 of the RtOSE pipeline: extracting the position of an object (a ball) from a recorded mobile video and converting that movement into physical distances and speeds.

## Overview

Pipeline (high level):

- Pre-recorded video
- Frame extraction
- Object segmentation (per frame)
- Centroid extraction
- Pixel displacement calculation
- Reference-based pixel → meter scaling
- Distance validation / smoothing
- Speed computation

## Inputs

- Recorded mobile video
- Known or extracted FPS (frames per second)
- A visible scene reference with known real-world length (for pixel → meter scaling)

## 1. Video input & frame extraction

- Read the video and extract frames in order. For Phase 1 do not skip frames unless explicitly required.
- Extract FPS from metadata when available; otherwise use a known value.

## 2. Object segmentation (per frame)

- For each frame compute a segmentation mask for the object (the ball). Using a mask instead of a bounding box reduces noise and gives more precise centroids, especially with partial occlusions.

Purpose:

- Avoid bounding-box noise
- Obtain accurate object shape and mask
- Improve centroid precision

## 3. Centroid extraction

From the segmentation mask M_i for frame i compute the centroid (using pixel moments). Using standard image moments:

$$x_i = \frac{M_{10}}{M_{00}}, \qquad y_i = \frac{M_{01}}{M_{00}}$$

where M_{pq} are the raw image moments computed on the binary mask M_i. The centroid is therefore

$$\text{centroid}(M_i) = (x_i, y_i)$$

This gives the object's pixel coordinates in frame i.

## 4. Pixel displacement calculation

Compute displacement between consecutive centroids (Euclidean distance in pixel space):

$$d_{pixels}(i) = \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}$$

This is the pixel distance traveled by the object between frames i and i+1.

## 5. Pixel → meter scaling (reference-based)

If the scene contains a visible reference object with known real-world length, measure its length in pixels (call it reference_pixels) and compute the scale:

$$\text{scale} = \frac{\text{real\\_distance\\_meters}}{\text{reference\\_pixels}}\quad\left(\frac{\text{meters}}{\text{pixel}}\right)$$

Convert pixel distances to meters:

$$d_{meters}(i) = d_{pixels}(i) \times \text{scale}$$

Notes:

- If the reference is not in the same plane as the object, perspective will introduce error. Prefer references in the same plane as the ball's motion.
- If multiple references exist, compute an average scale or use a homography if you want position-dependent scaling.

## 6. Distance validation & smoothing

Before computing speed, validate the distances and optionally smooth the trajectory to remove segmentation artifacts.

Checks and techniques:

- Spike detection / outlier rejection (e.g., remove points with sudden large jumps)
- Moving-average or median-filter smoothing
- Optional Kalman filter for combining prediction + measurement (helps with occlusions)

Edge cases to consider:

- Missing segmentation masks (object lost): interpolate or skip those frames
- Sudden large jumps due to wrong segmentation: detect and drop or smooth

## 7. Speed computation

Time between frames (assuming constant frame rate):

$$\Delta t = \frac{1}{\text{FPS}}$$

Instantaneous speed for frame i (using the distance between i and i+1):

$$v_i = \frac{d_{meters}(i)}{\Delta t}$$

Outputs you may want:

- Instantaneous speed series \\(v_i\\)
- Mean speed over the sequence
- Peak speed
- Window-averaged speed (sliding window smoothing)

## Implementation notes / contract

- Inputs: video file (with known or extractable FPS), scene reference length (meters)
- Outputs: per-frame centroid (px), per-frame distance (px and m), per-frame speed (m/s), summary statistics (mean, max)
- Error modes: missing masks, reference not visible, large perspective distortions

## Quick algorithmic checklist

1. Load video; extract FPS
2. For each frame:
   - run segmentation → mask M_i
   - compute centroid (x_i, y_i)
3. Compute d_pixels(i) between consecutive centroids
4. Compute scale from reference and convert to meters
5. Validate and smooth distances as needed
6. Compute instantaneous speeds and summary metrics

## How to preview

- Open `Phase1.md` in VS Code and use the Markdown preview (Cmd+Shift+V) to view formatted equations and sections.

## Next steps / improvements

- Add example code snippets (Python + OpenCV) for segmentation, centroid extraction and scaling.
- Add an example video and expected outputs for testing.
- Add unit tests for the math (moment calculations, scale conversion) and a small sample pipeline test.

---

If you want, I can also:

- add a small Python script demonstrating the full Phase 1 pipeline on a test frame sequence, or
- add a sample video and a minimal test harness.




