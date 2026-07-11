import numpy as np

# Unified keypoint maps to standardize landmarks across MediaPipe and YOLO Pose
KEYPOINT_MAPPING_MEDIAPIPE = {
    "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30, "left_foot_index": 31, "right_foot_index": 32
}

KEYPOINT_MAPPING_YOLO = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

class FeatureExtractor:
    @staticmethod
    def normalize_skeleton(landmarks, backend="mediapipe"):
        """
        Translates raw landmark indices into a unified coordinate dictionary.
        landmarks: numpy array of shape (N, 3) or (N, 4) [x, y, z, (visibility)]
        """
        if landmarks is None:
            return None
            
        mapping = KEYPOINT_MAPPING_MEDIAPIPE if backend.lower() == "mediapipe" else KEYPOINT_MAPPING_YOLO
        skeleton = {}
        
        for name, idx in mapping.items():
            if idx < len(landmarks):
                # Save x, y, z and optional visibility (default 1.0 if not present)
                pt = landmarks[idx]
                x = pt[0]
                y = pt[1]
                z = pt[2] if len(pt) > 2 else 0.0
                vis = pt[3] if len(pt) > 3 else 1.0
                skeleton[name] = np.array([x, y, z, vis])
                
        # Fill missing heel and foot index coordinates for YOLO Pose
        if "left_heel" not in skeleton and "left_ankle" in skeleton:
            skeleton["left_heel"] = skeleton["left_ankle"].copy()
        if "right_heel" not in skeleton and "right_ankle" in skeleton:
            skeleton["right_heel"] = skeleton["right_ankle"].copy()
        if "left_foot_index" not in skeleton and "left_ankle" in skeleton:
            skeleton["left_foot_index"] = skeleton["left_ankle"].copy()
        if "right_foot_index" not in skeleton and "right_ankle" in skeleton:
            skeleton["right_foot_index"] = skeleton["right_ankle"].copy()
            
        return skeleton

    @staticmethod
    def calculate_angle_3d(p1, p2, p3):
        """
        Calculates the 3D angle (in degrees) at joint p2 between segments p2-p1 and p2-p3.
        Points are numpy arrays of shape (3,) or (4,).
        """
        v1 = p1[:3] - p2[:3]
        v2 = p3[:3] - p2[:3]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 180.0
            
        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    @staticmethod
    def calculate_inclination(p1, p2, axis="vertical"):
        """
        Calculates inclination of segment p1-p2 relative to vertical (y-axis) or horizontal (x-axis).
        """
        v = p2[:3] - p1[:3]
        if axis == "vertical":
            ref = np.array([0.0, -1.0, 0.0]) # Screen vertical (y increases downwards)
        else:
            ref = np.array([1.0, 0.0, 0.0])  # Screen horizontal
            
        v_norm = v[:3] / (np.linalg.norm(v[:3]) + 1e-6)
        cosine_angle = np.dot(v_norm, ref)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    @classmethod
    def extract_features(cls, skeleton):
        """
        Extracts high-level body features from standard skeleton coordinate dict.
        Returns a dictionary of features and an ordered vector for ML models.
        """
        if skeleton is None:
            return None
            
        features = {}
        
        # 1. Core Joint Angles
        features["left_elbow"] = cls.calculate_angle_3d(skeleton["left_shoulder"], skeleton["left_elbow"], skeleton["left_wrist"])
        features["right_elbow"] = cls.calculate_angle_3d(skeleton["right_shoulder"], skeleton["right_elbow"], skeleton["right_wrist"])
        
        features["left_shoulder"] = cls.calculate_angle_3d(skeleton["left_elbow"], skeleton["left_shoulder"], skeleton["left_hip"])
        features["right_shoulder"] = cls.calculate_angle_3d(skeleton["right_elbow"], skeleton["right_shoulder"], skeleton["right_hip"])
        
        features["left_knee"] = cls.calculate_angle_3d(skeleton["left_hip"], skeleton["left_knee"], skeleton["left_ankle"])
        features["right_knee"] = cls.calculate_angle_3d(skeleton["right_hip"], skeleton["right_knee"], skeleton["right_ankle"])
        
        features["left_hip"] = cls.calculate_angle_3d(skeleton["left_shoulder"], skeleton["left_hip"], skeleton["left_knee"])
        features["right_hip"] = cls.calculate_angle_3d(skeleton["right_shoulder"], skeleton["right_hip"], skeleton["right_knee"])
        
        features["left_ankle"] = cls.calculate_angle_3d(skeleton["left_knee"], skeleton["left_ankle"], skeleton["left_foot_index"])
        features["right_ankle"] = cls.calculate_angle_3d(skeleton["right_knee"], skeleton["right_ankle"], skeleton["right_foot_index"])

        # 2. Spine and Body Inclination
        mid_shoulder = (skeleton["left_shoulder"][:3] + skeleton["right_shoulder"][:3]) / 2.0
        mid_hip = (skeleton["left_hip"][:3] + skeleton["right_hip"][:3]) / 2.0
        
        # Spine alignment: Neck to Mid-hip relative to vertical
        features["spine_alignment"] = cls.calculate_inclination(mid_hip, mid_shoulder, "vertical")
        features["back_inclination"] = cls.calculate_inclination(mid_hip, mid_shoulder, "vertical")
        
        # Hip alignment relative to horizontal
        features["hip_alignment"] = cls.calculate_inclination(skeleton["left_hip"], skeleton["right_hip"], "horizontal")
        features["shoulder_alignment"] = cls.calculate_inclination(skeleton["left_shoulder"], skeleton["right_shoulder"], "horizontal")
        
        # Neck tilt: Nose to mid-shoulder relative to vertical
        features["neck_tilt"] = cls.calculate_inclination(mid_shoulder, skeleton["nose"], "vertical")

        # 3. Spine Curvature Approximation (Angle: Nose -> Mid-Shoulder -> Mid-Hip)
        # Straight spine is close to 180 degrees.
        features["spine_curvature"] = cls.calculate_angle_3d(skeleton["nose"], mid_shoulder, mid_hip)

        # 4. Physical Dimensions & Ratios (Scale/Distance Normalization)
        shoulder_width = np.linalg.norm(skeleton["left_shoulder"][:3] - skeleton["right_shoulder"][:3])
        hip_width = np.linalg.norm(skeleton["left_hip"][:3] - skeleton["right_hip"][:3])
        features["shoulder_hip_ratio"] = shoulder_width / (hip_width + 1e-6)
        
        # Arm length to shoulder width ratios
        left_arm_len = (np.linalg.norm(skeleton["left_shoulder"][:3] - skeleton["left_elbow"][:3]) + 
                        np.linalg.norm(skeleton["left_elbow"][:3] - skeleton["left_wrist"][:3]))
        right_arm_len = (np.linalg.norm(skeleton["right_shoulder"][:3] - skeleton["right_elbow"][:3]) + 
                         np.linalg.norm(skeleton["right_elbow"][:3] - skeleton["right_wrist"][:3]))
        features["left_arm_to_shoulder"] = left_arm_len / (shoulder_width + 1e-6)
        features["right_arm_to_shoulder"] = right_arm_len / (shoulder_width + 1e-6)
        
        # Leg length to hip width ratios
        left_leg_len = (np.linalg.norm(skeleton["left_hip"][:3] - skeleton["left_knee"][:3]) + 
                        np.linalg.norm(skeleton["left_knee"][:3] - skeleton["left_ankle"][:3]))
        right_leg_len = (np.linalg.norm(skeleton["right_hip"][:3] - skeleton["right_knee"][:3]) + 
                         np.linalg.norm(skeleton["right_knee"][:3] - skeleton["right_ankle"][:3]))
        features["left_leg_to_hip"] = left_leg_len / (hip_width + 1e-6)
        features["right_leg_to_hip"] = right_leg_len / (hip_width + 1e-6)
        
        # Torso height
        torso_height = np.linalg.norm(mid_shoulder - mid_hip)
        features["torso_to_legs_ratio"] = torso_height / ((left_leg_len + right_leg_len)/2.0 + 1e-6)

        # 5. Anthropometric Center of Mass (CoM) Estimation
        # Weights partition: Torso 0.47, Head/Neck 0.08, Upper Arms 0.03x2, Forearms 0.02x2, Thighs 0.12x2, Shanks 0.055x2
        left_upper_arm = (skeleton["left_shoulder"][:3] + skeleton["left_elbow"][:3]) / 2.0
        right_upper_arm = (skeleton["right_shoulder"][:3] + skeleton["right_elbow"][:3]) / 2.0
        left_forearm = (skeleton["left_elbow"][:3] + skeleton["left_wrist"][:3]) / 2.0
        right_forearm = (skeleton["right_elbow"][:3] + skeleton["right_wrist"][:3]) / 2.0
        left_thigh = (skeleton["left_hip"][:3] + skeleton["left_knee"][:3]) / 2.0
        right_thigh = (skeleton["right_hip"][:3] + skeleton["right_knee"][:3]) / 2.0
        left_shank = (skeleton["left_knee"][:3] + skeleton["left_ankle"][:3]) / 2.0
        right_shank = (skeleton["right_knee"][:3] + skeleton["right_ankle"][:3]) / 2.0
        
        com = (
            0.470 * ((mid_shoulder + mid_hip) / 2.0) +
            0.080 * ((skeleton["nose"][:3] + mid_shoulder) / 2.0) +
            0.030 * left_upper_arm + 0.030 * right_upper_arm +
            0.020 * left_forearm + 0.020 * right_forearm +
            0.120 * left_thigh + 0.120 * right_thigh +
            0.055 * left_shank + 0.055 * right_shank
        )
        features["com_x"] = com[0]
        features["com_y"] = com[1]
        features["com_z"] = com[2]

        # 6. Balance Point (CoM projection relative to Base of Support - Feet midpoint)
        left_foot = skeleton["left_ankle"][:3]
        right_foot = skeleton["right_ankle"][:3]
        mid_feet = (left_foot + right_foot) / 2.0
        
        # Distance of CoM horizontal projection to feet midpoint
        features["balance_offset_x"] = float(com[0] - mid_feet[0])
        features["balance_offset_z"] = float(com[2] - mid_feet[2])
        # Support base is estimated as distance between feet
        feet_dist = np.linalg.norm(left_foot - right_foot)
        features["balance_stability"] = np.exp(-abs(features["balance_offset_x"]) / (feet_dist + 1e-6))

        # 7. Bilateral Symmetry Features
        features["elbow_symmetry"] = abs(features["left_elbow"] - features["right_elbow"])
        features["knee_symmetry"] = abs(features["left_knee"] - features["right_knee"])
        features["shoulder_symmetry"] = abs(features["left_shoulder"] - features["right_shoulder"])
        features["hip_symmetry"] = abs(features["left_hip"] - features["right_hip"])
        features["leg_length_symmetry"] = abs(left_leg_len - right_leg_len)

        # 8. Normalized Landmark Coordinates (for ML classifier model)
        # Shift skeleton origin to body center (mid-hip) and scale by torso height
        norm_factor = torso_height if torso_height > 1e-6 else 1.0
        norm_landmarks = {}
        for name, pt in skeleton.items():
            norm_pt = (pt[:3] - mid_hip) / norm_factor
            norm_landmarks[f"{name}_nx"] = norm_pt[0]
            norm_landmarks[f"{name}_ny"] = norm_pt[1]
            norm_landmarks[f"{name}_nz"] = norm_pt[2]
            
        features.update(norm_landmarks)
        
        return features

    @classmethod
    def get_feature_vector(cls, features):
        """
        Flattens features into an ordered vector for ML models (XGBoost/MLP).
        Uses a consistent key order.
        """
        if features is None:
            return None
            
        # Select list of feature keys to construct input vector (independent of normalized coords)
        keys = [
            "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
            "left_knee", "right_knee", "left_hip", "right_hip", "left_ankle", "right_ankle",
            "spine_alignment", "back_inclination", "hip_alignment", "shoulder_alignment",
            "neck_tilt", "spine_curvature", "shoulder_hip_ratio", "left_arm_to_shoulder",
            "right_arm_to_shoulder", "left_leg_to_hip", "right_leg_to_hip", "torso_to_legs_ratio",
            "balance_offset_x", "balance_stability", "elbow_symmetry", "knee_symmetry",
            "shoulder_symmetry", "hip_symmetry"
        ]
        
        # Add normalized landmark coordinates
        for name in KEYPOINT_MAPPING_MEDIAPIPE.keys():
            keys.extend([f"{name}_nx", f"{name}_ny", f"{name}_nz"])
            
        vector = []
        for k in keys:
            vector.append(features.get(k, 0.0))
            
        return np.array(vector, dtype=np.float32)
