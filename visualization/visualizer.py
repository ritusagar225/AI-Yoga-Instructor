import cv2
import numpy as np

# Standard skeletal connections to draw
SKELETON_CONNECTIONS = [
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Arms
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Legs
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle")
]

class PoseVisualizer:
    """
    Renders human skeletal overlays, joint angles, active error warnings,
    and a heads-up-display (HUD) directly onto video frames using OpenCV.
    """
    def __init__(self):
        # Premium color palette (BGR format)
        self.COLOR_GREEN = (46, 204, 113)    # Correct pose
        self.COLOR_YELLOW = (52, 152, 219)   # Low error (soft orange/yellow)
        self.COLOR_RED = (70, 70, 231)       # High error (neon red)
        self.COLOR_WHITE = (245, 245, 245)
        self.COLOR_GRAY = (128, 128, 128)
        self.COLOR_DARK_HUD = (35, 30, 30)   # Sleek overlay card

    def draw_pose(self, frame, skeleton, errors=None):
        """
        Draws the standard skeletal links and color-coded joint nodes.
        """
        if skeleton is None:
            return frame

        h, w, _ = frame.shape
        err_joints = {err["joint"]: err["severity"] for err in (errors or [])}

        # 1. Draw connections (bones)
        for start_joint, end_joint in SKELETON_CONNECTIONS:
            if start_joint in skeleton and end_joint in skeleton:
                pt1 = skeleton[start_joint]
                pt2 = skeleton[end_joint]
                
                # Convert normalized coords [0-1] to pixel coords
                x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
                x2, y2 = int(pt2[0] * w), int(pt2[1] * h)
                
                # Check if connection contains a joint with error
                color = self.COLOR_GREEN
                thickness = 2
                
                # If either connecting joint is flagged with error, color the bone
                for joint in [start_joint, end_joint]:
                    for err_name in err_joints.keys():
                        if err_name.startswith(joint):
                            severity = err_joints[err_name]
                            color = self.COLOR_RED if severity == "High" else self.COLOR_YELLOW
                            thickness = 3
                            break
                            
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # 2. Draw joint nodes and text labels
        for name, pt in skeleton.items():
            # Only draw structural joints
            if name in ["left_heel", "right_heel", "left_foot_index", "right_foot_index", "left_eye", "right_eye", "left_ear", "right_ear"]:
                continue
                
            x, y = int(pt[0] * w), int(pt[1] * h)
            
            # Determine node color
            color = self.COLOR_GREEN
            radius = 5
            
            for err_name, severity in err_joints.items():
                if err_name.startswith(name):
                    color = self.COLOR_RED if severity == "High" else self.COLOR_YELLOW
                    radius = 8
                    break
                    
            cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius + 2, self.COLOR_WHITE, 1, cv2.LINE_AA)

        return frame

    def draw_angles(self, frame, skeleton, features):
        """
        Draws current angle values adjacent to their respective joints.
        """
        if skeleton is None or features is None:
            return frame

        h, w, _ = frame.shape
        # Joint name -> landmark name to attach text label
        angle_labels = {
            "left_elbow": "left_elbow",
            "right_elbow": "right_elbow",
            "left_knee": "left_knee",
            "right_knee": "right_knee",
            "left_hip": "left_hip",
            "right_hip": "right_hip"
        }

        for joint_key, lm_key in angle_labels.items():
            if joint_key in features and lm_key in skeleton:
                angle_val = int(round(features[joint_key]))
                pt = skeleton[lm_key]
                x, y = int(pt[0] * w), int(pt[1] * h)
                
                # Offset text position to prevent overlap
                offset_x = 12 if "right" in lm_key else -55
                offset_y = -10
                
                cv2.putText(frame, f"{angle_val}deg", (x + offset_x, y + offset_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_WHITE, 1, cv2.LINE_AA)

        return frame

    def draw_hud(self, frame, pose_name, state_name, hold_time, score, feedback_tips):
        """
        Draws a modern HUD card at the top-left of the frame containing key statistics.
        """
        h, w, _ = frame.shape
        
        # 1. Draw HUD Background Overlay
        hud_w, hud_h = 320, 160
        sub_img = frame[15:15+hud_h, 15:15+hud_w]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 35
        # Blend background to get a dark semi-transparent glassmorphic look
        res = cv2.addWeighted(sub_img, 0.4, white_rect, 0.6, 1.0)
        frame[15:15+hud_h, 15:15+hud_w] = res
        cv2.rectangle(frame, (15, 15), (15 + hud_w, 15 + hud_h), self.COLOR_WHITE, 1, cv2.LINE_AA)

        # 2. Render Text Stats
        pose_title = pose_name.replace("_", " ").title() if pose_name else "Detecting..."
        cv2.putText(frame, pose_title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLOR_WHITE, 2, cv2.LINE_AA)
        
        # Color code state text
        state_color = self.COLOR_WHITE
        if "Holding" in state_name:
            state_color = self.COLOR_GREEN
        elif "Entering" in state_name:
            state_color = self.COLOR_YELLOW
            
        cv2.putText(frame, f"State: {state_name}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Hold Timer: {hold_time:.1f}s", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1, cv2.LINE_AA)
        
        # Draw dynamic score bar
        cv2.putText(frame, f"Posture Score: {score}%", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1, cv2.LINE_AA)
        bar_w = 200
        score_fill = int((score / 100.0) * bar_w)
        cv2.rectangle(frame, (30, 132), (30 + bar_w, 140), self.COLOR_GRAY, -1)
        
        bar_color = self.COLOR_GREEN if score >= 80 else (self.COLOR_YELLOW if score >= 60 else self.COLOR_RED)
        cv2.rectangle(frame, (30, 132), (30 + score_fill, 140), bar_color, -1)

        # 3. Draw Bottom Feedback Banner
        if feedback_tips:
            banner_h = 45
            banner_y = h - banner_h - 15
            
            sub_banner = frame[banner_y:banner_y+banner_h, 15:w-15]
            white_banner = np.ones(sub_banner.shape, dtype=np.uint8) * 35
            res_banner = cv2.addWeighted(sub_banner, 0.4, white_banner, 0.6, 1.0)
            frame[banner_y:banner_y+banner_h, 15:w-15] = res_banner
            cv2.rectangle(frame, (15, banner_y), (w-15, banner_y + banner_h), self.COLOR_WHITE, 1, cv2.LINE_AA)
            
            # Show top feedback tip
            top_tip = feedback_tips[0]["tip"]
            cv2.putText(frame, f"TIP: {top_tip}", (30, banner_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_YELLOW, 1, cv2.LINE_AA)

        return frame
