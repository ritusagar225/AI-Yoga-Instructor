import os
import sys
import time
import numpy as np
import cv2
import psutil

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose_detector import MediaPipePoseDetector, YOLOPoseDetector

def run_benchmark(num_frames=100):
    print("="*60)
    print("POSE DETECTION BENCHMARK SUITE")
    print("="*60)
    
    # 1. Prepare dummy frame (simulating standard 640x480 webcam input)
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Draw a simulated stick figure so detector finds a person
    cv2.circle(dummy_frame, (320, 150), 40, (255, 255, 255), -1)
    cv2.line(dummy_frame, (320, 190), (320, 350), (255, 255, 255), 5)
    cv2.line(dummy_frame, (320, 230), (260, 280), (255, 255, 255), 4)
    cv2.line(dummy_frame, (320, 230), (380, 280), (255, 255, 255), 4)
    cv2.line(dummy_frame, (320, 350), (280, 430), (255, 255, 255), 4)
    cv2.line(dummy_frame, (320, 350), (360, 430), (255, 255, 255), 4)

    results = {}
    
    # Get current process tracker
    process = psutil.Process(os.getpid())

    # 2. Benchmark MediaPipe
    print("\n[1/2] Benchmarking MediaPipe Pose...")
    try:
        mp_detector = MediaPipePoseDetector(model_complexity=1)
        
        # Warmup
        for _ in range(10):
            _ = mp_detector.process_frame(dummy_frame)
            
        # Benchmark run
        start_time = time.time()
        latencies = []
        cpu_usages = []
        mem_usages = []
        
        for i in range(num_frames):
            t0 = time.time()
            landmarks, conf = mp_detector.process_frame(dummy_frame)
            latencies.append((time.time() - t0) * 1000.0) # ms
            
            # Sample system stats every 10 frames
            if i % 10 == 0:
                cpu_usages.append(process.cpu_percent(interval=None))
                mem_usages.append(process.memory_info().rss / (1024.0 * 1024.0)) # MB
                
        total_time = time.time() - start_time
        
        results["MediaPipe"] = {
            "fps": num_frames / total_time,
            "latency_avg": np.mean(latencies),
            "latency_std": np.std(latencies),
            "cpu_avg": np.mean(cpu_usages) if cpu_usages else 0.0,
            "mem_avg": np.mean(mem_usages) if mem_usages else 0.0,
            "dims": "3D Landmarks (33 points)"
        }
        print(f"MediaPipe Completed: {results['MediaPipe']['fps']:.2f} FPS | Latency: {results['MediaPipe']['latency_avg']:.2f}ms")
    except Exception as e:
        print(f"MediaPipe Benchmark Failed: {e}")
        results["MediaPipe"] = None

    # 3. Benchmark YOLOv8 Pose
    print("\n[2/2] Benchmarking YOLOv8 Pose...")
    try:
        # Load small/fast nano pose model
        yolo_detector = YOLOPoseDetector(model_name="yolov8n-pose.pt", conf=0.3, device="cpu")
        
        # Warmup
        for _ in range(10):
            _ = yolo_detector.process_frame(dummy_frame)
            
        # Benchmark run
        start_time = time.time()
        latencies = []
        cpu_usages = []
        mem_usages = []
        
        for i in range(num_frames):
            t0 = time.time()
            landmarks, conf = yolo_detector.process_frame(dummy_frame)
            latencies.append((time.time() - t0) * 1000.0)
            
            if i % 10 == 0:
                cpu_usages.append(process.cpu_percent(interval=None))
                mem_usages.append(process.memory_info().rss / (1024.0 * 1024.0))
                
        total_time = time.time() - start_time
        
        results["YOLOv8 Pose"] = {
            "fps": num_frames / total_time,
            "latency_avg": np.mean(latencies),
            "latency_std": np.std(latencies),
            "cpu_avg": np.mean(cpu_usages) if cpu_usages else 0.0,
            "mem_avg": np.mean(mem_usages) if mem_usages else 0.0,
            "dims": "2D Landmarks (17 points)"
        }
        print(f"YOLOv8 Completed: {results['YOLOv8 Pose']['fps']:.2f} FPS | Latency: {results['YOLOv8 Pose']['latency_avg']:.2f}ms")
    except Exception as e:
        print(f"YOLOv8 Benchmark Failed: {e}")
        results["YOLOv8 Pose"] = None

    # 4. Generate Markdown Report
    generate_report(results)

def generate_report(results):
    report_dir = "docs"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "benchmark_report.md")
    
    mp = results.get("MediaPipe")
    yolo = results.get("YOLOv8 Pose")
    
    mp_str = f"| MediaPipe Pose | {mp['fps']:.1f} | {mp['latency_avg']:.1f}ms (±{mp['latency_std']:.1f}ms) | {mp['cpu_avg']:.1f}% | {mp['mem_avg']:.1f} MB | {mp['dims']} |" if mp else "| MediaPipe Pose | N/A | N/A | N/A | N/A | N/A |"
    yolo_str = f"| YOLOv8 Pose | {yolo['fps']:.1f} | {yolo['latency_avg']:.1f}ms (±{yolo['latency_std']:.1f}ms) | {yolo['cpu_avg']:.1f}% | {yolo['mem_avg']:.1f} MB | {yolo['dims']} |" if yolo else "| YOLOv8 Pose | N/A | N/A | N/A | N/A | N/A |"

    md_content = f"""# Pose Estimation Backend Benchmarking Report

This report compares the performance, resource requirements, and trade-offs of the MediaPipe Pose and YOLOv8-pose backends profiled on an identical standard video stream simulation.

---

## 1. Quantitative Performance Analysis

| Backend Model | Throughput (FPS) | Inference Latency (ms) | CPU Load (%) | RAM Usage (MB) | Keypoint Dimensions |
| :--- | :---: | :---: | :---: | :---: | :---: |
{mp_str}
{yolo_str}

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
"""

    with open(report_path, "w") as f:
        f.write(md_content)
        
    print(f"\nBenchmark report successfully generated at: {report_path}")

if __name__ == "__main__":
    run_benchmark()
