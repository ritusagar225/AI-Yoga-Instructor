import time
from enum import Enum

class PoseState(Enum):
    NO_PERSON = "No Person Detected"
    ENTERING = "Entering Pose"
    HOLDING = "Holding Pose"
    EXITING = "Exiting Pose"

class PoseStateMachine:
    """
    State machine that manages yoga pose transitions, holding states, and timers.
    Prevents hold-timer manipulation by validating alignment quality thresholds.
    """
    def __init__(self, alignment_threshold=0.80, hold_stabilize_time=1.0):
        self.alignment_threshold = alignment_threshold       # Score above which pose is considered "Held"
        self.hold_stabilize_time = hold_stabilize_time       # Seconds of stable pose before entering HOLDING
        
        self.state = PoseState.NO_PERSON
        self.current_pose = None
        self.state_enter_time = time.time()
        
        self.hold_start_time = None
        self.hold_duration = 0.0
        self.session_holds = {} # Stores total hold times per pose in the current session

    def reset_timer(self):
        self.hold_start_time = None
        self.hold_duration = 0.0

    def update(self, detected_pose, alignment_score, person_detected, frame_confidence):
        """
        Updates the state machine based on the current frame statistics.
        detected_pose: Name of the classified pose (str) or None.
        alignment_score: Normalised similarity score [0-1.0].
        person_detected: Boolean indicator.
        frame_confidence: Average tracking confidence of landmarks.
        
        Returns:
            state: The current PoseState
            hold_duration: Active holding time in seconds
            state_duration: Time spent in the current state
        """
        now = time.time()
        
        # 1. Handle No Person Detection or Low Tracking Confidence
        if not person_detected or frame_confidence < 0.55 or detected_pose is None:
            if self.state != PoseState.NO_PERSON:
                # Save hold time if we were holding a pose
                if self.state == PoseState.HOLDING and self.current_pose:
                    self._save_hold_duration()
                self.state = PoseState.NO_PERSON
                self.current_pose = None
                self.state_enter_time = now
                self.reset_timer()
            return self.state, 0.0, now - self.state_enter_time

        # If a person is detected but pose is new, initialize tracking
        if self.current_pose != detected_pose:
            if self.state == PoseState.HOLDING and self.current_pose:
                self._save_hold_duration()
            self.current_pose = detected_pose
            self.state = PoseState.ENTERING
            self.state_enter_time = now
            self.reset_timer()

        state_duration = now - self.state_enter_time

        # 2. Evaluate State Transitions
        if self.state == PoseState.ENTERING:
            if alignment_score >= self.alignment_threshold:
                # If alignment is good, check if we've stabilized long enough
                if self.hold_start_time is None:
                    self.hold_start_time = now
                elif now - self.hold_start_time >= self.hold_stabilize_time:
                    # Stably aligned for longer than stabilize limit, transition to HOLDING
                    self.state = PoseState.HOLDING
                    self.state_enter_time = now
                    self.hold_start_time = now - self.hold_stabilize_time # Back-date to when alignment started
            else:
                # Reset transition timer if alignment drops below threshold
                self.hold_start_time = None
                
        elif self.state == PoseState.HOLDING:
            if alignment_score < (self.alignment_threshold - 0.05): # Hysteresis window to prevent rapid toggling
                # User's alignment slipped, transition to EXITING
                self._save_hold_duration()
                self.state = PoseState.EXITING
                self.state_enter_time = now
                self.reset_timer()
            else:
                # Maintain hold state, compute active duration
                self.hold_duration = now - self.hold_start_time
                
        elif self.state == PoseState.EXITING:
            if alignment_score >= self.alignment_threshold:
                # Re-entered pose correctly
                self.state = PoseState.HOLDING
                self.state_enter_time = now
                self.hold_start_time = now
            elif state_duration > 2.0:
                # Spent too long outside the pose, revert to entering
                self.state = PoseState.ENTERING
                self.state_enter_time = now
                self.reset_timer()

        return self.state, self.hold_duration, now - self.state_enter_time

    def _save_hold_duration(self):
        """
        Saves the completed pose hold duration to session logs.
        """
        if self.current_pose and self.hold_duration > 0.5:
            prev_hold = self.session_holds.get(self.current_pose, 0.0)
            self.session_holds[self.current_pose] = prev_hold + self.hold_duration
            
    def get_session_analytics(self):
        """
        Returns a dictionary of total hold times for all poses in this session.
        """
        # Save any currently running hold time before returning analytics
        if self.state == PoseState.HOLDING and self.current_pose:
            now = time.time()
            active_duration = now - self.hold_start_time
            prev_hold = self.session_holds.get(self.current_pose, 0.0)
            # Temporary view of total hold time including active hold
            total_holds = self.session_holds.copy()
            total_holds[self.current_pose] = prev_hold + active_duration
            return total_holds
            
        return self.session_holds
