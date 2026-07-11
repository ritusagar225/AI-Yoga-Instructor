import numpy as np
import time

class OneEuroFilter:
    """
    Adaptive low-pass filter designed for human landmark coordinates to reduce jitter.
    Adjusts cutoff frequency based on target velocity.
    """
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = np.array(x0, dtype=np.float32)
        self.dx_prev = np.zeros_like(x0, dtype=np.float32)
        self.t_prev = t0

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        t = float(t)
        x = np.array(x, dtype=np.float32)
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
            
        # Calculate speed (derivative)
        dx = (x - self.x_prev) / dt
        
        # Filter derivative to reduce noise in speed estimation
        d_alpha = self._alpha(self.d_cutoff, dt)
        dx_hat = d_alpha * dx + (1.0 - d_alpha) * self.dx_prev
        
        # Calculate dynamic cutoff frequency
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        
        # Filter target signal
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        
        # Save state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat


class SkeletalOneEuroFilter:
    """
    Wrapper of OneEuroFilter for multiple keypoints (e.g. 33 landmarks for MediaPipe).
    Supports dynamic shapes and individual joint filters.
    """
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.filters = {}
        self.init_time = time.time()

    def filter_skeleton(self, landmarks):
        """
        landmarks: numpy array of shape (N, 3) or (N, 4) containing coords.
        Returns: smoothed landmarks of same shape.
        """
        if landmarks is None:
            return None
            
        t = time.time() - self.init_time
        smoothed = np.copy(landmarks)
        
        for idx, pt in enumerate(landmarks):
            if idx not in self.filters:
                self.filters[idx] = OneEuroFilter(t, pt, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
            smoothed[idx] = self.filters[idx](t, pt)
            
        return smoothed


class SimpleKalmanFilter:
    """
    A 1D/Multi-D Constant Velocity Kalman Filter to smooth joint positions.
    """
    def __init__(self, state_dim=3, process_noise=1e-3, measurement_noise=1e-1):
        self.state_dim = state_dim
        # State vector: [x, vx, y, vy, z, vz...]
        self.x = None
        self.P = np.eye(2 * state_dim) * 1.0
        self.Q = np.eye(2 * state_dim) * process_noise
        self.R = np.eye(state_dim) * measurement_noise
        self.H = np.zeros((state_dim, 2 * state_dim))
        for i in range(state_dim):
            self.H[i, 2*i] = 1.0
            
    def predict(self, dt):
        if self.x is None:
            return
        # Transition matrix F
        F = np.eye(2 * self.state_dim)
        for i in range(self.state_dim):
            F[2*i, 2*i + 1] = dt
            
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        z = np.array(z)
        if self.x is None:
            self.x = np.zeros(2 * self.state_dim)
            for i in range(self.state_dim):
                self.x[2*i] = z[i]
            return self.x[::2]
            
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2 * self.state_dim) - K @ self.H) @ self.P
        return self.x[::2]

    def filter(self, z, dt=0.033):
        self.predict(dt)
        return self.update(z)


class SkeletalKalmanFilter:
    """
    Wrapper for Kalmans across N landmarks.
    """
    def __init__(self, process_noise=1e-4, measurement_noise=5e-2):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.filters = {}
        self.last_time = None

    def filter_skeleton(self, landmarks):
        if landmarks is None:
            return None
            
        now = time.time()
        dt = now - self.last_time if self.last_time is not None else 0.033
        self.last_time = now
        
        smoothed = np.copy(landmarks)
        for idx, pt in enumerate(landmarks):
            dim = len(pt)
            if idx not in self.filters:
                self.filters[idx] = SimpleKalmanFilter(state_dim=dim, process_noise=self.process_noise, measurement_noise=self.measurement_noise)
            smoothed[idx] = self.filters[idx].filter(pt, dt)
            
        return smoothed


class ExponentialMovingAverageFilter:
    """
    Simple Exponential Moving Average filter.
    Formula: y_t = alpha * x_t + (1 - alpha) * y_{t-1}
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.state = None

    def filter_skeleton(self, landmarks):
        if landmarks is None:
            return None
        if self.state is None:
            self.state = np.copy(landmarks)
            return self.state
            
        self.state = self.alpha * landmarks + (1.0 - self.alpha) * self.state
        return self.state
