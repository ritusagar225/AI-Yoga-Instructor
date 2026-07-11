import os
import json
import numpy as np

# Definitive list of the 23 supported yoga poses
POSE_CLASSES = [
    "mountain_pose", "tree_pose", "warrior_i", "warrior_ii", "warrior_iii",
    "triangle_pose", "extended_side_angle", "chair_pose", "easy_pose", "childs_pose",
    "butterfly_pose", "staff_pose", "seated_forward_bend", "cobra", "bridge",
    "camel", "wheel", "crow_pose", "boat_pose", "side_plank",
    "downward_dog", "headstand", "shoulder_stand"
]

class PoseMLP:
    """
    Lightweight PyTorch MLP for classifying yoga poses based on extracted features.
    """
    def __init__(self, input_dim=91, num_classes=23):
        # We import torch dynamically so the rest of the application remains functional
        # even if PyTorch is still being installed in the background.
        import torch
        import torch.nn as nn
        
        class MLPNet(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, out_dim)
                )
            def forward(self, x):
                return self.net(x)
                
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MLPNet(input_dim, num_classes).to(self.device)
        self.model.eval()

    def load_weights(self, path):
        import torch
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            return True
        return False

    def predict(self, feature_vector, temperature=1.2):
        """
        Runs model inference and returns calibrated softmax probabilities.
        temperature: Scaling factor for confidence calibration (T > 1.0 smooths overconfidence).
        """
        import torch
        import torch.nn.functional as F
        
        x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            # Temperature scaling for confidence calibration
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=1).cpu().numpy()[0]
            
        return probs


class YogaPoseClassifier:
    """
    High-level yoga pose classifier. Integrates PyTorch MLP with a 
    fully-featured geometric heuristic fallback classifier.
    """
    def __init__(self, weights_path="weights/pose_mlp.pth", config_path="config/poses_config.json"):
        self.weights_path = weights_path
        self.config_path = config_path
        self.mlp = None
        self.poses_config = {}
        
        # Load configurations
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.poses_config = json.load(f).get("poses", {})

        # Attempt to initialize MLP
        try:
            self.mlp = PoseMLP()
            self.mlp_loaded = self.mlp.load_weights(weights_path)
        except Exception:
            self.mlp_loaded = False

    def classify(self, feature_vector, features_dict):
        """
        Classifies the yoga pose using the MLP model if loaded,
        otherwise falls back to rule-based geometric heuristics.
        
        Returns:
            pose_name: str (e.g. "tree_pose")
            confidence: float (calibrated probability)
            top_3: list of tuples (pose_name, prob)
        """
        if self.mlp_loaded and feature_vector is not None:
            try:
                probs = self.mlp.predict(feature_vector)
                top_indices = np.argsort(probs)[::-1][:3]
                
                top_3 = [(POSE_CLASSES[idx], float(probs[idx])) for idx in top_indices]
                pose_name = top_3[0][0]
                confidence = top_3[0][1]
                return pose_name, confidence, top_3
            except Exception:
                pass # Fallback to heuristics on error

        return self.classify_heuristics(features_dict)

    def classify_heuristics(self, features_dict):
        """
        Fallback heuristic classifier.
        Compares joint angles against targets defined in the poses config.
        """
        if not self.poses_config or features_dict is None:
            # Absolute fallback if config is missing
            return "mountain_pose", 1.0, [("mountain_pose", 1.0)]

        pose_scores = []
        for pose_name, config in self.poses_config.items():
            joints_config = config.get("joints", {})
            
            # Compute total deviation score
            deviation_sum = 0.0
            weight_sum = 0.0
            
            for joint_name, params in joints_config.items():
                if joint_name in features_dict:
                    val = features_dict[joint_name]
                    mean = params["mean"]
                    std = params["std"]
                    weight = params["weight"]
                    
                    # Compute normalized z-score deviation
                    z = abs(val - mean) / (std + 1e-6)
                    deviation_sum += z * weight
                    weight_sum += weight
                    
            if weight_sum > 0:
                avg_deviation = deviation_sum / weight_sum
                # Convert deviation into a probability-like score using an exponential decay
                score = np.exp(-avg_deviation / 3.0)
                # Penalize partial feature matches
                matched_count = sum(1 for j in joints_config if j in features_dict)
                match_fraction = matched_count / len(joints_config) if len(joints_config) > 0 else 0.0
                score *= match_fraction
            else:
                score = 0.0
                
            pose_scores.append((pose_name, score))
            
        # Normalize scores to sum to 1.0 for probability simulation
        scores_arr = np.array([s[1] for s in pose_scores])
        sum_scores = np.sum(scores_arr)
        if sum_scores > 0:
            scores_arr = scores_arr / sum_scores
        else:
            scores_arr = np.ones_like(scores_arr) / len(scores_arr)
            
        calibrated_poses = [(pose_scores[i][0], float(scores_arr[i])) for i in range(len(pose_scores))]
        calibrated_poses.sort(key=lambda x: x[1], reverse=True)
        
        top_3 = calibrated_poses[:3]
        return top_3[0][0], top_3[0][1], top_3
