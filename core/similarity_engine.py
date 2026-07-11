import json
import os
import numpy as np

class SimilarityEngine:
    """
    Evaluates pose similarity and alignment against ideal statistical configurations.
    Uses joint angle deviations, coordinate Euclidean distances, and cosine similarities.
    """
    def __init__(self, config_path="config/poses_config.json", reference_dir="data/reference"):
        self.config_path = config_path
        self.reference_dir = reference_dir
        self.poses_config = {}
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.poses_config = json.load(f).get("poses", {})

    def _get_reference_skeleton(self, pose_name):
        """
        Loads reference normalized coordinates from a JSON file if available.
        Otherwise, returns a default standing skeleton configuration.
        """
        ref_path = os.path.join(self.reference_dir, f"{pose_name}.json")
        if os.path.exists(ref_path):
            with open(ref_path, "r") as f:
                return json.load(f)
        
        # Fallback template coordinate offsets (centered at mid-hip, scaled by torso height)
        # Suitable for general cosine similarity comparison
        return {
            "left_shoulder": [-0.5, 1.0, 0.0],
            "right_shoulder": [0.5, 1.0, 0.0],
            "left_hip": [-0.3, 0.0, 0.0],
            "right_hip": [0.3, 0.0, 0.0],
            "left_knee": [-0.3, -1.0, 0.0],
            "right_knee": [0.3, -1.0, 0.0],
            "left_ankle": [-0.3, -2.0, 0.0],
            "right_ankle": [0.3, -2.0, 0.0],
            "left_elbow": [-1.0, 1.0, 0.0],
            "right_elbow": [1.0, 1.0, 0.0],
            "left_wrist": [-1.5, 1.0, 0.0],
            "right_wrist": [1.5, 1.0, 0.0]
        }

    def compute_similarity(self, pose_name, features_dict, skeleton_dict):
        """
        Compares the current features and landmarks against the ideal target pose.
        
        Returns:
            similarity_score: float (0 to 1.0) based on normalized landmark coordinates
            alignment_score: float (0 to 1.0) based on joint angle configurations
            confidence: float (average tracking confidence of joints used)
            overall_score: float (0 to 1.0)
        """
        if pose_name not in self.poses_config or features_dict is None:
            return 0.0, 0.0, 0.0, 0.0

        config = self.poses_config[pose_name]
        joints_config = config.get("joints", {})

        # 1. Compute Joint Angle Alignment Score
        # Matches user's joint angles with mean angles, penalized by standard deviations
        total_error = 0.0
        weight_sum = 0.0
        
        for joint, params in joints_config.items():
            if joint in features_dict:
                user_val = features_dict[joint]
                mean_val = params["mean"]
                std_val = params["std"]
                tolerance = params["tolerance"]
                weight = params["weight"]
                
                # Deviation from mean. Allow tolerance offset for free.
                dev = max(0.0, abs(user_val - mean_val) - tolerance)
                
                # Scale error relative to std dev (Z-score penalty)
                joint_error = dev / (std_val + 1e-6)
                total_error += joint_error * weight
                weight_sum += weight

        if weight_sum > 0:
            avg_error = total_error / weight_sum
            # Scale alignment between 0 and 1.0. A z-deviation average of 3.0 results in ~37% score.
            alignment_score = float(np.exp(-avg_error / 3.0))
        else:
            alignment_score = 0.0

        # 2. Compute Coordinate-based Cosine Similarity
        # Flatten current coordinates and reference coordinates into vectors
        ref_skeleton = self._get_reference_skeleton(pose_name)
        user_vec = []
        ref_vec = []
        
        # Check tracking confidence for used points
        confidences = []
        
        for key in ref_skeleton.keys():
            if key in skeleton_dict:
                user_pt = skeleton_dict[key] # [x, y, z, vis]
                ref_pt = ref_skeleton[key]   # [x, y, z]
                
                user_vec.extend(user_pt[:3])
                ref_vec.extend(ref_pt[:3])
                confidences.append(user_pt[3]) # joint visibility score
                
        user_vec = np.array(user_vec)
        ref_vec = np.array(ref_vec)
        
        if len(user_vec) > 0 and np.linalg.norm(user_vec) > 0 and np.linalg.norm(ref_vec) > 0:
            # Cosine similarity range is [-1.0, 1.0] -> normalize to [0, 1.0]
            dot_prod = np.dot(user_vec, ref_vec)
            norm_prod = np.linalg.norm(user_vec) * np.linalg.norm(ref_vec)
            cosine_sim = float((dot_prod / norm_prod + 1.0) / 2.0)
            
            # Euclidean distance (Procrustes shape distance)
            euclidean_dist = float(np.linalg.norm(user_vec - ref_vec) / np.sqrt(len(user_vec)))
            dist_score = float(np.exp(-euclidean_dist))
            
            # Blend cosine and distance similarities
            similarity_score = 0.6 * cosine_sim + 0.4 * dist_score
        else:
            similarity_score = 0.0
            
        avg_confidence = float(np.mean(confidences)) if len(confidences) > 0 else 0.0

        # 3. Overall Score Blend
        # Overall is a weighted product of alignment (angles) and similarity (landmarks)
        overall_score = 0.7 * alignment_score + 0.3 * similarity_score
        
        return similarity_score, alignment_score, avg_confidence, overall_score
