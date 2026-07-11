import cv2
import numpy as np

class BasePoseDetector:
    """
    Abstract interface for human pose detectors.
    """
    def __init__(self, **kwargs):
        pass

    def process_frame(self, frame):
        """
        Processes a single frame.
        frame: OpenCV image (BGR).
        Returns:
            landmarks: numpy array of shape (N, 4) where each row is [x, y, z, visibility]
            confidence: overall frame detection score
        """
        raise NotImplementedError("process_frame must be implemented by subclasses.")


class MediaPipePoseDetector(BasePoseDetector):
    """
    MediaPipe Pose estimation wrapper.
    """
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, **kwargs):
        super().__init__()
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None, 0.0
            
        h, w, _ = frame.shape
        landmarks = []
        
        # Extract the 33 MediaPipe landmark coordinates
        for lm in results.pose_landmarks.landmark:
            # MediaPipe normalized coordinates (0-1) are converted to pixel coordinates
            # but we preserve normalized coordinates for ML pipeline, while keeping z scale.
            # We return: [x, y, z, visibility]
            landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
            
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Calculate frame-level tracking confidence as average visibility of critical joints
        critical_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] # shoulders, elbows, wrists, hips, knees, ankles
        if len(landmarks) > max(critical_joints):
            avg_confidence = float(np.mean(landmarks[critical_joints, 3]))
        else:
            avg_confidence = float(np.mean(landmarks[:, 3]))
            
        return landmarks, avg_confidence


class YOLOPoseDetector(BasePoseDetector):
    """
    YOLOv8-Pose estimation wrapper using Ultralytics.
    """
    def __init__(self, model_name="yolov8n-pose.pt", conf=0.5, device=None, **kwargs):
        super().__init__()
        from ultralytics import YOLO
        import torch
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load YOLO model (downloads weights dynamically if not present locally)
        self.model = YOLO(model_name)
        self.conf = conf

    def process_frame(self, frame):
        # Run inference
        results = self.model(frame, verbose=False, conf=self.conf, device=self.device)
        
        if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints.data) == 0:
            return None, 0.0
            
        # Select the keypoints of the first detected person (index 0)
        # YOLO keypoints shape: (num_persons, 17, 3) where last dimension is [x, y, conf]
        kp_tensor = results[0].keypoints.data[0] # (17, 3)
        kp = kp_tensor.cpu().numpy()
        
        h, w, _ = frame.shape
        landmarks = []
        
        # Convert pixel coordinates from YOLO to normalized [0-1] coordinates to match MediaPipe
        for pt in kp:
            x_norm = pt[0] / w
            y_norm = pt[1] / h
            z_norm = 0.0 # YOLO pose does not output 3D z-depth coordinates, default to 0.0
            conf = pt[2]
            landmarks.append([x_norm, y_norm, z_norm, conf])
            
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Compute overall confidence
        avg_confidence = float(np.mean(landmarks[:, 3]))
        
        return landmarks, avg_confidence
