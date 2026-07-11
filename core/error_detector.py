import json
import os
from enum import Enum

class Severity(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class ErrorDetector:
    """
    Scans joint angles against statistical bounds to flag incorrect body alignment.
    Categorizes errors by severity and calculates exact angular deviations.
    """
    def __init__(self, config_path="config/poses_config.json"):
        self.config_path = config_path
        self.poses_config = {}
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.poses_config = json.load(f).get("poses", {})

    def detect_errors(self, pose_name, features_dict):
        """
        Scans current joint angles and flags deviations that exceed target tolerances.
        
        Returns:
            errors: list of dicts. Each dict contains:
                {
                    "joint": str,
                    "user_val": float,
                    "target_mean": float,
                    "deviation": float,
                    "direction": str ("under" or "over"),
                    "severity": str ("Low", "Medium", "High"),
                    "correction_tip": str
                }
        """
        errors = []
        if pose_name not in self.poses_config or features_dict is None:
            return errors

        config = self.poses_config[pose_name]
        joints_config = config.get("joints", {})

        for joint_name, params in joints_config.items():
            if joint_name not in features_dict:
                continue

            user_val = features_dict[joint_name]
            mean_val = params["mean"]
            std_val = params["std"]
            tolerance = params["tolerance"]
            weight = params["weight"]

            raw_dev = user_val - mean_val
            abs_dev = abs(raw_dev)

            # Check if deviation exceeds allowed tolerance
            if abs_dev > tolerance:
                z_score = (abs_dev - tolerance) / (std_val + 1e-6)
                
                # Determine Severity level
                if z_score < 1.5:
                    severity = Severity.LOW.value
                elif z_score < 3.0:
                    severity = Severity.MEDIUM.value
                else:
                    severity = Severity.HIGH.value

                direction = "over" if raw_dev > 0 else "under"
                deviation_deg = abs_dev
                
                # Generate natural correction recommendations
                tip = self._generate_correction_tip(joint_name, mean_val, user_val, direction, abs_dev)
                
                errors.append({
                    "joint": joint_name,
                    "user_val": float(user_val),
                    "target_mean": float(mean_val),
                    "deviation": float(deviation_deg),
                    "direction": direction,
                    "severity": severity,
                    "weight": weight,
                    "correction_tip": tip
                })

        # Sort errors by priority: weight * deviation magnitude, bringing high-impact errors to front
        errors.sort(key=lambda x: x["weight"] * x["deviation"], reverse=True)
        return errors

    def _generate_correction_tip(self, joint_name, mean_val, user_val, direction, deviation):
        """
        Creates context-aware, directionally correct guidelines for the user.
        """
        joint_display = joint_name.replace("_", " ").title()
        dev_round = int(round(deviation))

        # Joint specific message routing
        if "knee" in joint_name:
            if mean_val > 150: # Straight leg target (e.g. Mountain, Warrior back leg)
                return f"Straighten your {joint_display.replace('Knee', 'knee')}. Lift out of your joint."
            else: # Bent knee target (e.g. Chair, Warrior front leg)
                if direction == "over":
                    return f"Straighten your {joint_display.replace('Knee', 'knee')} slightly by {dev_round}°."
                else:
                    return f"Bend your {joint_display.replace('Knee', 'knee')} deeper by {dev_round}°."
                    
        elif "elbow" in joint_name:
            if mean_val > 150: # Straight arms
                return f"Straighten your {joint_display.replace('Elbow', 'elbow')}."
            else:
                if direction == "over":
                    return f"Straighten your {joint_display.replace('Elbow', 'elbow')} slightly by {dev_round}°."
                else:
                    return f"Bend your {joint_display.replace('Elbow', 'elbow')} more by {dev_round}°."

        elif "shoulder" in joint_name:
            if "alignment" in joint_name:
                return "Level your shoulders. Relax them away from your ears."
            if direction == "under":
                return f"Open your chest and raise your {joint_display.replace('Shoulder', 'arms')} higher by {dev_round}°."
            else:
                return f"Lower your {joint_display.replace('Shoulder', 'arms')} slightly by {dev_round}°."

        elif "hip" in joint_name:
            if "alignment" in joint_name:
                return "Square your hips to the front."
            if direction == "under":
                return f"Sink your hips lower by {dev_round}°."
            else:
                return f"Raise your hips slightly by {dev_round}°."

        elif "spine_alignment" in joint_name or "back_inclination" in joint_name:
            if mean_val < 15: # Vertically straight spine
                return "Lengthen your spine. Stand tall and pull your navel in."
            elif direction == "under":
                return f"Hinge forward more from your hips by {dev_round}°."
            else:
                return f"Lift your torso slightly, reduce forward bend by {dev_round}°."

        elif "spine_curvature" in joint_name:
            return "Release rounding in your back. Lengthen your spine from tailbone to crown."

        elif "neck_tilt" in joint_name:
            return "Align your neck. Look straight ahead and lift your chin slightly."

        return f"Adjust {joint_display} configuration by about {dev_round}°."
