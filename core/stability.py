from collections import deque
import numpy as np

class StabilityAnalyzer:
    """
    Evaluates postural stability by tracking temporal variance, body sway,
    and frame-to-frame joint speed across a sliding window of frames.
    """
    def __init__(self, window_size=30):
        self.window_size = window_size
        # Store coordinate history of key anchor joints: shoulders, hips, knees
        self.history = deque(maxlen=window_size)
        self.anchor_joints = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee"]

    def reset(self):
        self.history.clear()

    def update(self, skeleton_dict):
        """
        Updates coordinate history and computes a stability score.
        
        Returns:
            stability_score: float [0, 1.0] (1.0 indicates perfect stability/stillness)
            sway_magnitude: float (average variance of joints)
        """
        if skeleton_dict is None:
            return 0.0, 0.0

        # Extract coordinates of anchor joints
        current_anchors = []
        for joint in self.anchor_joints:
            if joint in skeleton_dict:
                current_anchors.append(skeleton_dict[joint][:3]) # x, y, z
                
        if len(current_anchors) < len(self.anchor_joints):
            # Not all anchor joints tracked
            return 0.0, 0.0

        current_anchors = np.array(current_anchors) # Shape: (6, 3)
        self.history.append(current_anchors)

        if len(self.history) < 5:
            # Not enough frames yet for temporal analysis
            return 1.0, 0.0

        # 1. Compute Frame-to-Frame Velocity (Speed)
        # Average distance traveled by joints between consecutive frames
        speeds = []
        for i in range(1, len(self.history)):
            disp = np.linalg.norm(self.history[i] - self.history[i-1], axis=1) # (6,)
            speeds.append(np.mean(disp))
            
        mean_speed = float(np.mean(speeds))

        # 2. Compute Coordinate Variance (Drift)
        # Standard deviation of each joint over the sliding window
        history_arr = np.array(self.history) # Shape: (W, 6, 3)
        # Variance along the time axis (W)
        variance_per_joint = np.var(history_arr, axis=0) # Shape: (6, 3)
        mean_variance = float(np.mean(variance_per_joint))

        # 3. Calculate Stability Score
        # Map speed and variance to [0, 1.0] using exponential decay
        # Sensitivity coefficients: 15.0 for speed, 120.0 for variance (calibrated for normalized coordinates)
        speed_penalty = mean_speed * 20.0
        variance_penalty = mean_variance * 150.0
        
        stability_score = float(np.exp(-(speed_penalty + variance_penalty)))
        stability_score = np.clip(stability_score, 0.0, 1.0)
        
        # Sway magnitude represents raw coordinate deviation
        sway_magnitude = float(np.sqrt(mean_variance))

        return stability_score, sway_magnitude
