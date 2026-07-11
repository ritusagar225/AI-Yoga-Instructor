# Pose Estimation Backend Benchmarking Report

This report compares the performance, resource requirements, and trade-offs of the MediaPipe Pose and YOLOv8-pose backends profiled on an identical standard video stream simulation.

---

## 1. Quantitative Performance Analysis

| Backend Model | Throughput (FPS) | Inference Latency (ms) | CPU Load (%) | RAM Usage (MB) | Keypoint Dimensions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| MediaPipe Pose | 34.8 | 28.7ms (±3.5ms) | 82.0% | 489.5 MB | 3D Landmarks (33 points) |
| YOLOv8 Pose | 14.3 | 70.0ms (±5.8ms) | 683.4% | 776.3 MB | 2D Landmarks (17 points) |

---

## 2. Qualitative Backend Trade-offs

### MediaPipe Pose
* **Strengths**: 
  * Exceptionally fast, optimized for CPU execution on mobile/edge.
  * Provides native 3D camera depth estimation landmarks ($z$-coordinate).
  * Returns 33 keypoints, including detailed face, hand, and heel landmarks (critical for stance/weight distribution).
* **Weaknesses**:
  * Susceptible to high noise/jitter under motion.
  * Struggles with body occlusions and complex overlaps.

### YOLOv8 Pose (Nano)
* **Strengths**:
  * High spatial accuracy and highly robust against occlusions/overlaps.
  * Captures body structure reliably even in low-light or angled viewpoints.
* **Weaknesses**:
  * Demands significantly more memory and CPU cycles; requires dedicated GPU (CUDA/TensorRT) for high-framerate real-time rendering.
  * Returns only 2D COCO keypoints (17 points), requiring template depth estimation for 3D projections.

---

## 3. Engineering Recommendations

* **Web-Browser/Real-time Deployment**: MediaPipe is the recommended default. It runs smoothly on standard laptops without GPU accelerators, and its 3D depth parameters enable accurate 3D joint angle computations.
* **Heavy Server-Side Analytics / Occlusion Scenarios**: YOLOv8-pose is the ideal choice for post-session processing, complex multi-person workouts, or extreme camera positions, provided GPU hardware acceleration is available.
