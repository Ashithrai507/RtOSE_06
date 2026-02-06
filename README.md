# RtOSE_06
## Real Time Object Speed Estimation
##  Project Overview

This project aims to estimate the **real-world speed of moving objects** (such as a cricket ball, baseball, or car) using a **single camera feed** — either from a **mobile phone camera** or a **pre-recorded video**.  
The system leverages **computer vision (CV)** and **machine learning** techniques to track object motion frame-by-frame and convert pixel displacement into real-world distance, enabling near real-time speed calculation.

The core idea follows a simple physical principle:

> **Speed = Distance / Time**

However, achieving accurate results requires solving challenges such as camera calibration, perspective distortion, frame timing, and object tracking stability.

---
## Objectives
<details>
 <summary></summary>


- Detect a moving object in a video stream
- Track the object across consecutive frames
- Convert pixel movement into real-world distance
- Calculate object speed with high accuracy
- Support both **live camera input** and **recorded videos**
- Build a scalable pipeline that can later be deployed on mobile devices

---

##  Key Concept

A video provides two crucial elements:
- **Time information** (frames per second)
- **Spatial information** (pixel coordinates)

The missing link is **mapping pixels to real-world distance**, which is solved using calibration or reference-based techniques.
</details>
---

##  System Architecture 
```sscs
Video Input (Camera / File)
↓
Object Detection
↓
Object Tracking
↓
Pixel-to-World Distance Mapping
↓
Displacement Calculation
↓
Time Calculation (FPS-based)
↓
Speed Estimation
```


---
##  Methodology
<details>




### 1. Video Acquisition
- Input can be:
  - Mobile phone camera (real-time)
  - Pre-recorded video
- FPS is extracted to calculate time intervals accurately.

---

### 2. Object Detection
The system identifies the object of interest in each frame using:
- Deep learning-based detectors
- Outputs bounding box or object centroid `(x, y)`.

---

### 3. Object Tracking
Tracking ensures consistent object identity across frames and smooth motion estimation.
- Techniques used:
  - Kalman Filter
  - SORT / DeepSORT
  - Optical Flow (for high-speed motion)

---

### 4. Pixel-to-Real-World Mapping
This is the most critical step.

Supported approaches:
- **Reference-based scaling**
  - Uses known distances (pitch length, lane width, etc.)
- **Camera calibration + homography**
  - Maps image plane to ground plane
- **Monocular depth estimation (optional)**
  - AI-based depth prediction (less accurate but flexible)

---

### 5. Distance Calculation
- Measure frame-to-frame displacement in pixels
- Convert pixel displacement to meters using calibration data
- Accumulate total distance traveled

---

### 6. Time Calculation
- Time per frame = `1 / FPS`
- Total time = `number_of_frames / FPS`

---

### 7. Speed Estimation
Speed is calculated using:
 **Speed = Distance / Time**


- Enhancements:
- Moving average smoothing
- Velocity estimation over multiple frames
- Noise reduction using Kalman filtering

---
</details>





## Phase 1:
<details>


## Method 1: Reference Object Scaling 
Calculate the real time distance of the object between the camera 
### Pixel-to-Real-World Distance Estimation Using Known Reference

---

##  Overview

Method 1 estimates real-world distance by using an object in the video frame whose **actual physical length is known**.  
By measuring the pixel length of this reference object, we compute a **pixel-to-meter scale factor**, which allows conversion of any pixel displacement into real-world distance.

This method is widely used in:
- Sports analytics (cricket, baseball)
- Traffic speed monitoring
- Fixed-camera surveillance systems

---

## Core Principle

A camera captures images in **pixels**, not meters.  
If a known object appears in the same plane as the moving object, it can act as a **ruler** for the entire scene.

Once the scale is known:
**Real Distance (meters) = Pixel Distance × Scale**

---

## 📐 Mathematical Foundation
### Euclidean Distance in Pixel Space
Given two points in the image:
```sscs
P1 = (x1, y1)
P2 = (x2, y2)

Pixel distance is computed as:
```

---

### Pixel-to-Meter Scale Factor

If the real-world distance of the reference object is known:
This gives:

**1 pixel = scale meters**

---

## 🏗️ Step-by-Step Methodology

---

### Step 1: Select a Reference Object

The reference object must:
- Have a known real-world length
- Be clearly visible in the video frame
- Lie on the same plane as the moving object

#### Common Examples

| Application | Reference Object | Real Length |
|-----------|-----------------|-------------|
| Cricket | Pitch length | 20.12 m |
| Baseball | Base distance | 27.4 m |
| Cars | Lane width | 3.7 m |
| Indoor scenes | Floor tiles / court lines | Known |

---

### Step 2: Mark Reference Points

Identify two endpoints of the reference object in the image:

```json
R1 = (x1, y1)
R2 = (x2, y2)
```

These points can be:
- Manually selected (recommended for prototypes)
- Automatically detected (advanced stage)

---

### Step 3: Compute Reference Pixel Distance
**reference_pixels = √((x2 − x1)² + (y2 − y1)²)**
This gives the reference length in pixels.
---
### Step 4: Compute Scale Factor
**scale = reference_real_length_meters / reference_pixels**
```json
Example:
Pitch length = 20.12 m
Pixel length = 980 px
scale = 20.12 / 980 ≈ 0.0205 m/px

```

---
### Step 5: Detect the Moving Object
For each frame, detect the object and extract its centroid:
Object position at frame i → (xi, yi)
Only the centroid is needed for distance estimation.

---
### Step 6: Compute Object Pixel Displacement
Between two consecutive frames:
**object_pixels = √((xi+1 − xi)² + (yi+1 − yi)²)**

---
### Step 7: Convert Pixel Distance to Real-World Distance
**object_distance_meters = object_pixels × scale**
This represents the real-world distance traveled between frames.
---
##  Example Calculation
- Video FPS: 240
- Scale: 0.0205 m/px
- Ball displacement: 45 px
- Distance = 45 × 0.0205 = 0.92 meters


---

## ⚠️ Assumptions & Constraints

| Assumption | Reason |
|----------|--------|
| Fixed camera | Prevents scale variation |
| Same motion plane | Avoids perspective distortion |
| High FPS | Improves temporal accuracy |
| Minimal camera tilt | Reduces depth error |

Violating these assumptions will reduce accuracy.

---

## ❌ Common Errors

- Using object height instead of ground reference
- Mixing depth planes (air vs ground)
- Handheld or shaky camera footage
- Ignoring perspective shortening

---

## ✅ When This Method Works Best

- Cricket ball speed estimation
- Baseball pitch analysis
- Vehicle speed on straight roads
- Athletic performance tracking

---

## 🔥 Advantages

- Simple and fast to implement
- No camera calibration required
- No depth estimation needed
- High accuracy under controlled conditions

---

## 📌 Limitations

- Requires known reference distance
- Sensitive to camera movement
- Assumes planar motion

---

## 📍 Conclusion

Reference Object Scaling is the **most reliable starting point** for real-world distance estimation using a single camera.  
It forms the foundation for speed calculation, trajectory analysis, and advanced motion modeling in computer vision systems.

Once implemented correctly, this method enables accurate speed estimation with minimal computational overhead.

---

## ➡️ Next Steps

- Temporal smoothing of distance measurements
- Speed estimation using FPS
- Error correction using Kalman filtering
- Extension to homography-based mapping (Method 2)

</details>

## More details
<details>
## Expected Accuracy

| Scenario                                | Accuracy |
|----------------------------------------|----------|
| Fixed camera + known reference distance | ±3–5%    |
| Calibrated camera + homography          | ±1–3%    |
| Handheld camera without calibration    | ±8–12%   |
| Depth-estimation-only approach          | Variable |

---

##  Challenges & Limitations

- Motion blur for fast-moving objects
- Rolling shutter distortion in mobile cameras
- Perspective distortion
- Camera shake in handheld recording
- Occlusion of the object
- Low frame rate limiting precision

---

##  Tech Stack

- Programming Language: **Python**
- Computer Vision: **:contentReference[oaicite:0]{index=0}**
- Object Detection: YOLO (v8 or later)
- Tracking: DeepSORT / Kalman Filter
- Math & Processing: NumPy
- Optional Mobile Deployment:
  - Android CameraX
  - TensorFlow Lite / ONNX

---

##  Applications

- Cricket and baseball ball speed estimation
- Vehicle speed detection (traffic monitoring)
- Sports analytics
- Training and performance analysis
- Smart surveillance systems


---

##  Conclusion

This project demonstrates that **accurate real-world speed estimation is achievable using a single camera** when combined with robust computer vision techniques and proper calibration.  
The approach bridges classical physics and modern AI, making it suitable for both academic research and real-world deployment.

---

### Author
**Ashith Rai**  
B.Tech – Artificial Intelligence & Machine Learning


