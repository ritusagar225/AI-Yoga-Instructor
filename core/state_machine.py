import time
from enum import Enum


class PoseState(Enum):
    NO_PERSON = "No Person Detected"
    ENTERING = "Entering Pose"
    HOLDING = "Holding Pose"
    EXITING = "Exiting Pose"


class PoseStateMachine:
    """
    State machine for yoga pose transitions.

    States:
    NO_PERSON -> ENTERING -> HOLDING -> EXITING

    The machine waits for a pose to remain stable before starting the
    hold timer and uses hysteresis to avoid rapid state switching.
    """

    def __init__(
        self,
        alignment_threshold=0.60,
        hold_stabilize_time=0.8
    ):
        # Minimum alignment score required to hold a pose
        self.alignment_threshold = alignment_threshold

        # Time the pose must remain stable before HOLDING
        self.hold_stabilize_time = hold_stabilize_time

        self.state = PoseState.NO_PERSON
        self.current_pose = None

        self.state_enter_time = time.time()

        self.hold_start_time = None
        self.hold_duration = 0.0

        # Total hold time for every pose
        self.session_holds = {}

    def reset_timer(self):
        self.hold_start_time = None
        self.hold_duration = 0.0

    def update(
        self,
        detected_pose,
        alignment_score,
        person_detected,
        frame_confidence,
    ):
        """
        Updates the pose state.

        Parameters
        ----------
        detected_pose : str | None
        alignment_score : float
        person_detected : bool
        frame_confidence : float

        Returns
        -------
        (state, hold_duration, state_duration)
        """

        now = time.time()

        # -------------------------------------------------
        # No person detected
        # -------------------------------------------------
        if (
            not person_detected
            or detected_pose is None
            or frame_confidence < 0.45
        ):

            if self.state != PoseState.NO_PERSON:

                if (
                    self.state == PoseState.HOLDING
                    and self.current_pose
                ):
                    self._save_hold_duration()

                self.state = PoseState.NO_PERSON
                self.current_pose = None
                self.state_enter_time = now
                self.reset_timer()

            return (
                self.state,
                0.0,
                now - self.state_enter_time,
            )

        # -------------------------------------------------
        # New pose detected
        # -------------------------------------------------
        if self.current_pose != detected_pose:

            if (
                self.state == PoseState.HOLDING
                and self.current_pose
            ):
                self._save_hold_duration()

            self.current_pose = detected_pose
            self.state = PoseState.ENTERING
            self.state_enter_time = now
            self.reset_timer()

        state_duration = now - self.state_enter_time

        # -------------------------------------------------
        # ENTERING
        # -------------------------------------------------
        if self.state == PoseState.ENTERING:

            if alignment_score >= self.alignment_threshold:

                if self.hold_start_time is None:

                    self.hold_start_time = now

                elif (
                    now - self.hold_start_time
                    >= self.hold_stabilize_time
                ):

                    self.state = PoseState.HOLDING
                    self.state_enter_time = now

                    # Start hold timer from first stable frame
                    self.hold_start_time = (
                        now - self.hold_stabilize_time
                    )

            else:

                self.hold_start_time = None

        # -------------------------------------------------
        # HOLDING
        # -------------------------------------------------
        elif self.state == PoseState.HOLDING:

            if alignment_score < (
                self.alignment_threshold - 0.05
            ):

                self._save_hold_duration()

                self.state = PoseState.EXITING
                self.state_enter_time = now
                self.reset_timer()

            else:

                self.hold_duration = (
                    now - self.hold_start_time
                )

        # -------------------------------------------------
        # EXITING
        # -------------------------------------------------
        elif self.state == PoseState.EXITING:

            if alignment_score >= self.alignment_threshold:

                self.state = PoseState.HOLDING
                self.state_enter_time = now
                self.hold_start_time = now

            elif state_duration > 1.0:

                self.state = PoseState.ENTERING
                self.state_enter_time = now
                self.reset_timer()

        return (
            self.state,
            self.hold_duration,
            now - self.state_enter_time,
        )

    def _save_hold_duration(self):
        """Save completed hold duration."""

        if self.current_pose and self.hold_duration > 0.5:

            previous = self.session_holds.get(
                self.current_pose,
                0.0,
            )

            self.session_holds[self.current_pose] = (
                previous + self.hold_duration
            )

    def get_session_analytics(self):
        """Return cumulative hold times."""

        if (
            self.state == PoseState.HOLDING
            and self.current_pose
        ):

            now = time.time()

            active_duration = (
                now - self.hold_start_time
            )

            total = self.session_holds.copy()

            total[self.current_pose] = (
                total.get(self.current_pose, 0.0)
                + active_duration
            )

            return total

        return self.session_holds