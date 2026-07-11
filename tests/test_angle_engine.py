import numpy as np
import pytest
from core.feature_extractor import FeatureExtractor

def test_calculate_angle_3d_orthogonal():
    """
    Test 3D angle calculation for orthogonal vectors (90 degrees).
    """
    p1 = np.array([1.0, 0.0, 0.0]) # x-axis unit
    p2 = np.array([0.0, 0.0, 0.0]) # origin (joint)
    p3 = np.array([0.0, 1.0, 0.0]) # y-axis unit
    
    angle = FeatureExtractor.calculate_angle_3d(p1, p2, p3)
    assert pytest.approx(angle, 0.01) == 90.0

def test_calculate_angle_3d_straight():
    """
    Test 3D angle calculation for colinear opposite vectors (180 degrees).
    """
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([-1.0, 0.0, 0.0])
    
    angle = FeatureExtractor.calculate_angle_3d(p1, p2, p3)
    assert pytest.approx(angle, 0.01) == 180.0

def test_calculate_angle_3d_acute():
    """
    Test 3D angle calculation for an equilateral triangle angle (60 degrees).
    """
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([0.5, np.sqrt(3)/2.0, 0.0]) # cos(60) = 0.5, sin(60) = sqrt(3)/2
    
    angle = FeatureExtractor.calculate_angle_3d(p1, p2, p3)
    assert pytest.approx(angle, 0.01) == 60.0

def test_calculate_inclination_vertical():
    """
    Test segment inclination angle relative to screen vertical.
    """
    p1 = np.array([0.0, 0.0, 0.0])
    # Segment points straight up (same direction as vertical reference)
    p2 = np.array([0.0, -1.0, 0.0]) 
    
    # pointing up is 0 degrees difference from vertical reference
    angle = FeatureExtractor.calculate_inclination(p1, p2, axis="vertical")
    assert abs(angle) < 0.1

    # Segment points sideways (90 degrees to vertical)
    p3 = np.array([1.0, 0.0, 0.0])
    angle2 = FeatureExtractor.calculate_inclination(p1, p3, axis="vertical")
    assert pytest.approx(angle2, 0.01) == 90.0
