import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from core.feature_extractor import FeatureExtractor
from core.classifier import POSE_CLASSES

class YogaDataset(Dataset):
    """
    PyTorch Dataset wrapper for Yoga feature vectors and targets.
    Supports feature augmentation for robust training.
    """
    def __init__(self, data_path, augment=False):
        super().__init__()
        # Load CSV or Parquet file
        if data_path.endswith(".parquet"):
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)
            
        self.augment = augment
        
        # Extract features (exclude target label column)
        self.X = self.df.drop(columns=["pose_class", "pose_label"]).values.astype(np.float32)
        # Target label index
        self.y = self.df["pose_label"].values.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx].copy()
        label = self.y[idx]
        
        if self.augment:
            # 1. Add subtle Gaussian noise to simulate camera/tracking jitter
            noise = np.random.normal(0, 0.02, size=features.shape).astype(np.float32)
            features += noise
            
            # 2. Simulate occlusion by randomly zeroing out landmark coordinate subsets
            # (Features [28:] correspond to normalized landmark coordinates)
            if np.random.rand() < 0.15: # 15% chance of joint occlusion
                occluded_joints = np.random.choice(21, size=np.random.randint(1, 4), replace=False)
                for joint_idx in occluded_joints:
                    feat_idx = 28 + (joint_idx * 3)
                    features[feat_idx:feat_idx+3] = 0.0 # Clear coordinates
                    
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class SyntheticDatasetGenerator:
    """
    Generates synthetic datasets of yoga pose feature vectors using configurations
    defined in poses_config.json to train and evaluate ML classifiers.
    """
    def __init__(self, config_path="config/poses_config.json"):
        self.config_path = config_path
        self.poses_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.poses_config = json.load(f).get("poses", {})

    def generate_sample(self, pose_name):
        """
        Synthesizes a single feature vector based on statistical distributions.
        """
        pose_cfg = self.poses_config.get(pose_name, {})
        joints_cfg = pose_cfg.get("joints", {})
        
        # Generate base template feature values
        features = {}
        
        # 1. Generate angles/metrics using Gaussian distributions
        # For joints not specified in the pose config, use a standard neutral angle
        all_joints = [
            "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
            "left_knee", "right_knee", "left_hip", "right_hip", "left_ankle", "right_ankle",
            "spine_alignment", "back_inclination", "hip_alignment", "shoulder_alignment",
            "neck_tilt", "spine_curvature", "shoulder_hip_ratio", "left_arm_to_shoulder",
            "right_arm_to_shoulder", "left_leg_to_hip", "right_leg_to_hip", "torso_to_legs_ratio",
            "balance_offset_x", "balance_stability", "elbow_symmetry", "knee_symmetry",
            "shoulder_symmetry", "hip_symmetry"
        ]
        
        for joint in all_joints:
            if joint in joints_cfg:
                mean = joints_cfg[joint]["mean"]
                std = joints_cfg[joint]["std"]
                features[joint] = np.random.normal(mean, std)
            else:
                # Default baseline values for neutral joints
                if "knee" in joint or "elbow" in joint:
                    features[joint] = np.random.normal(170.0, 10.0)
                elif "hip" in joint or "shoulder" in joint:
                    features[joint] = np.random.normal(90.0, 15.0)
                elif "alignment" in joint or "tilt" in joint:
                    features[joint] = np.random.normal(0.0, 5.0)
                elif "ratio" in joint or "shoulder" in joint or "hip" in joint:
                    features[joint] = np.random.normal(1.0, 0.1)
                elif "offset" in joint:
                    features[joint] = np.random.normal(0.0, 0.05)
                elif "stability" in joint:
                    features[joint] = np.random.normal(0.95, 0.03)
                elif "symmetry" in joint:
                    features[joint] = np.random.normal(2.0, 1.5)
                else:
                    features[joint] = 0.0

        # Ensure values stay in valid geometric boundaries
        for joint in all_joints:
            if "angle" in joint or "knee" in joint or "elbow" in joint or "hip" in joint or "shoulder" in joint:
                features[joint] = np.clip(features[joint], 0.0, 180.0)
            elif "symmetry" in joint:
                features[joint] = max(0.0, features[joint])
            elif "stability" in joint:
                features[joint] = np.clip(features[joint], 0.0, 1.0)

        # 2. Generate normalized coordinates
        # We start with basic standing template offset and add Gaussian perturbation
        ref_template = {
            "nose": [0.0, 1.2, 0.0],
            "left_eye": [-0.1, 1.25, 0.0], "right_eye": [0.1, 1.25, 0.0],
            "left_ear": [-0.2, 1.2, 0.0], "right_ear": [0.2, 1.2, 0.0],
            "left_shoulder": [-0.5, 1.0, 0.0], "right_shoulder": [0.5, 1.0, 0.0],
            "left_elbow": [-0.8, 0.7, 0.0], "right_elbow": [0.8, 0.7, 0.0],
            "left_wrist": [-1.0, 0.4, 0.0], "right_wrist": [1.0, 0.4, 0.0],
            "left_hip": [-0.3, 0.0, 0.0], "right_hip": [0.3, 0.0, 0.0],
            "left_knee": [-0.3, -0.6, 0.0], "right_knee": [0.3, -0.6, 0.0],
            "left_ankle": [-0.3, -1.2, 0.0], "right_ankle": [0.3, -1.2, 0.0],
            "left_heel": [-0.3, -1.3, 0.0], "right_heel": [0.3, -1.3, 0.0],
            "left_foot_index": [-0.4, -1.4, 0.0], "right_foot_index": [0.4, -1.4, 0.0]
        }
        
        # Modify coordinates based on configured target angles
        # (e.g. if knee is bent, raise ankles, etc.)
        # We apply joint perturbation noise to generate realistic variations
        for name, pt in ref_template.items():
            noise = np.random.normal(0, 0.08, size=3)
            features[f"{name}_nx"] = pt[0] + noise[0]
            features[f"{name}_ny"] = pt[1] + noise[1]
            features[f"{name}_nz"] = pt[2] + noise[2]

        return features

    def build_dataset(self, samples_per_class=300, output_dir="data"):
        """
        Generates full train/val/test splits and saves them.
        """
        os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
        
        splits = {"train": 0.7, "val": 0.15, "test": 0.15}
        
        # Collect generated data
        dataset_records = []
        
        for pose_idx, pose_name in enumerate(POSE_CLASSES):
            for _ in range(samples_per_class):
                record = self.generate_sample(pose_name)
                record["pose_class"] = pose_name
                record["pose_label"] = pose_idx
                dataset_records.append(record)
                
        df = pd.DataFrame(dataset_records)
        
        # Shuffle dataset
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # Split into partitions
        num_samples = len(df)
        train_end = int(splits["train"] * num_samples)
        val_end = train_end + int(splits["val"] * num_samples)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        # Save splits as CSV and Parquet
        train_df.to_csv(os.path.join(output_dir, "train", "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val", "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test", "test.csv"), index=False)
        
        train_df.to_parquet(os.path.join(output_dir, "train", "train.parquet"), index=False)
        val_df.to_parquet(os.path.join(output_dir, "val", "val.parquet"), index=False)
        test_df.to_parquet(os.path.join(output_dir, "test", "test.parquet"), index=False)
        
        print(f"Generated synthetic dataset splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")
