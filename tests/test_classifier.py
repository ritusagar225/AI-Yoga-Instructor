import pytest
from core.classifier import YogaPoseClassifier

def test_heuristic_classifier_mountain_pose():
    """
    Verifies that the heuristic classifier correctly classifies a Mountain Pose
    when joint features represent straight limbs and spine.
    """
    classifier = YogaPoseClassifier(config_path="config/poses_config.json")
    
    # Mock Mountain Pose features
    mock_features = {
        "left_knee": 178.0,
        "right_knee": 179.0,
        "left_hip": 177.0,
        "right_hip": 178.0,
        "left_elbow": 175.0,
        "right_elbow": 176.0,
        "spine_alignment": 2.0,
        "shoulder_alignment": 0.5
    }
    
    pose_name, confidence, top_3 = classifier.classify(None, mock_features)
    
    assert pose_name == "mountain_pose"
    assert confidence > 0.0
    assert top_3[0][0] == "mountain_pose"

def test_heuristic_classifier_tree_pose():
    """
    Verifies that the heuristic classifier correctly identifies Tree Pose
    when one leg is bent at 45 degrees and the other is straight.
    """
    classifier = YogaPoseClassifier(config_path="config/poses_config.json")
    
    # Mock Tree Pose features (using the pose config joint criteria)
    mock_features = {
        "standing_leg_knee": 178.0,
        "bent_leg_knee": 46.0,
        "bent_leg_hip": 88.0,
        "left_elbow": 92.0,
        "right_elbow": 88.0,
        "spine_alignment": 1.5,
        "shoulder_alignment": 1.0,
        # Standardize for general check
        "left_knee": 178.0,
        "right_knee": 46.0
    }
    
    pose_name, confidence, top_3 = classifier.classify(None, mock_features)
    
    assert pose_name == "tree_pose"
    assert top_3[0][0] == "tree_pose"
