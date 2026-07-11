import numpy as np

# List of poses where left and right side configurations should be identical
SYMMETRIC_POSES = [
    "mountain_pose", "chair_pose", "easy_pose", "childs_pose", "butterfly_pose",
    "staff_pose", "cobra", "bridge", "camel", "wheel", "downward_dog",
    "headstand", "shoulder_stand"
]

class SymmetryAnalyzer:
    """
    Evaluates bilateral skeletal symmetry.
    Customizes evaluations based on whether a pose is symmetric or asymmetric.
    """
    def __init__(self):
        pass

    def evaluate_symmetry(self, pose_name, features_dict):
        """
        Computes bilateral symmetry scores based on joint angles and line tilts.
        
        Returns:
            symmetry_score: float [0, 1.0] (1.0 indicates perfect symmetry)
            symmetry_breakdown: dict containing raw differences
        """
        if features_dict is None:
            return 0.0, {}

        # 1. Extract raw bilateral differences
        elbow_diff = features_dict.get("elbow_symmetry", 0.0)
        knee_diff = features_dict.get("knee_symmetry", 0.0)
        shoulder_diff = features_dict.get("shoulder_symmetry", 0.0)
        hip_diff = features_dict.get("hip_symmetry", 0.0)
        
        shoulder_tilt = features_dict.get("shoulder_alignment", 0.0)
        hip_tilt = features_dict.get("hip_alignment", 0.0)

        breakdown = {
            "elbow_diff": float(elbow_diff),
            "knee_diff": float(knee_diff),
            "shoulder_diff": float(shoulder_diff),
            "hip_diff": float(hip_diff),
            "shoulder_tilt": float(shoulder_tilt),
            "hip_tilt": float(hip_tilt)
        }

        # 2. Score computation based on pose type
        if pose_name in SYMMETRIC_POSES:
            # For symmetric poses, joint configurations must match left-to-right
            # Devs above 15 degrees reduce symmetry score significantly
            joint_errors = (elbow_diff + knee_diff + shoulder_diff + hip_diff) / 4.0
            tilt_errors = (shoulder_tilt + hip_tilt) / 2.0
            
            # Weighted penalty: 70% joint angle matches, 30% horizontal levelness
            penalty = (joint_errors * 0.04) + (tilt_errors * 0.06)
            symmetry_score = float(np.exp(-penalty))
        else:
            # For asymmetric poses (like Warrior or Tree), limbs are intentionally asymmetric.
            # We measure alignment levelness of the structural chassis: shoulders and hips
            tilt_errors = (shoulder_tilt + hip_tilt) / 2.0
            # Higher tilt penalty coefficient because horizontal leveling is critical
            penalty = tilt_errors * 0.08
            symmetry_score = float(np.exp(-penalty))

        symmetry_score = np.clip(symmetry_score, 0.0, 1.0)
        return symmetry_score, breakdown
