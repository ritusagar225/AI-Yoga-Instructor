import numpy as np
import pytest
import time
from core.filter import OneEuroFilter, SkeletalOneEuroFilter

def test_one_euro_filter_smoothing():
    """
    Tests that the One Euro Filter reduces high-frequency coordinate noise.
    """
    # Instantiate filter at t=0, coordinate=0.0
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    filt = OneEuroFilter(t0, x0, min_cutoff=1.0, beta=0.01)
    
    # Send a noisy jump at t=0.1
    t1 = 0.1
    x1 = np.array([10.0, 0.0, 0.0], dtype=np.float32) # massive sudden jump
    
    x_filtered = filt(t1, x1)
    
    # The filtered output should be smoothed, meaning it shouldn't jump all the way to 10.0
    assert x_filtered[0] < 8.0
    assert x_filtered[0] > 0.0 # but it should move in that direction
    
    # Send subsequent frames to show convergence
    t = t1
    for _ in range(5):
        t += 0.1
        x_filtered = filt(t, x1)
        
    # Should converge closer to the target 10.0 after multiple stable frames
    assert x_filtered[0] > 8.0

def test_skeletal_one_euro_filter():
    """
    Verifies that the Skeletal wrapper filters multiple joints correctly.
    """
    sk_filt = SkeletalOneEuroFilter(min_cutoff=1.0, beta=0.01)
    
    # Mock skeleton coordinates for 2 joints
    frame1 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0]
    ], dtype=np.float32)
    
    frame2 = np.array([
        [0.1, 0.0, 0.0],
        [2.0, 1.0, 1.0] # joint 2 has a large jump
    ], dtype=np.float32)
    
    f1_filtered = sk_filt.filter_skeleton(frame1)
    time.sleep(0.05)
    f2_filtered = sk_filt.filter_skeleton(frame2)
    
    # Joint 1 should move slightly
    assert abs(f2_filtered[0, 0] - 0.1) < 0.1
    
    # Joint 2's large jump (from 1.0 to 2.0) should be smoothed
    assert f2_filtered[1, 0] < 2.0
    assert f2_filtered[1, 0] > 1.0
