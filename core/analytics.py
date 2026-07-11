import os
import json
import time
from collections import Counter
import pandas as pd

class SessionAnalytics:
    """
    Collects, aggregates, and serializes yoga training session metrics.
    Saves frame logs in CSV/Parquet and summary reports in JSON.
    """
    def __init__(self, output_dir="data/processed"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.frame_data = []
        self.pose_holds = {}

    def log_frame(self, pose_name, scores_dict, errors_list, state_name):
        """
        Logs metrics for a single frame.
        scores_dict: {"alignment", "symmetry", "stability", "overall"}
        errors_list: list of active errors from ErrorDetector
        """
        active_errors = [err["joint"] for err in errors_list]
        
        self.frame_data.append({
            "timestamp": time.time() - self.start_time,
            "pose_name": pose_name,
            "state": state_name,
            "overall_score": scores_dict.get("overall", 0),
            "alignment_score": scores_dict.get("alignment", 0),
            "symmetry_score": scores_dict.get("symmetry", 0),
            "stability_score": scores_dict.get("stability", 0),
            "active_errors": ",".join(active_errors),
            "num_errors": len(active_errors)
        })

    def save_session(self, pose_holds=None):
        """
        Aggregates session logs, computes summary statistics, and saves output files.
        """
        if not self.frame_data:
            return None
            
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # Convert frame logs to DataFrame
        df = pd.DataFrame(self.frame_data)
        
        # 1. Compute Aggregates
        avg_overall = float(df["overall_score"].mean())
        avg_alignment = float(df["alignment_score"].mean())
        avg_symmetry = float(df["symmetry_score"].mean())
        avg_stability = float(df["stability_score"].mean())
        
        # Filter active poses (exclude NO_PERSON and ENTERING frames for best/weakest)
        active_df = df[~df["pose_name"].isna()]
        
        best_pose = None
        weakest_pose = None
        
        if not active_df.empty:
            pose_groups = active_df.groupby("pose_name")["overall_score"].mean()
            if not pose_groups.empty:
                best_pose = str(pose_groups.idxmax())
                weakest_pose = str(pose_groups.idxmin())

        # Count most frequent errors
        all_errors = []
        for err_str in df["active_errors"]:
            if err_str:
                all_errors.extend(err_str.split(","))
                
        error_counts = Counter(all_errors)
        most_frequent_errors = [
            {"joint": joint, "count": count} 
            for joint, count in error_counts.most_common(5)
        ]

        # Use provided state machine hold times or approximate them
        holds = pose_holds if pose_holds is not None else {}
        
        # 2. Compile Summary Report
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        summary = {
            "session_id": timestamp_str,
            "total_practice_time_seconds": float(total_time),
            "average_overall_score": float(round(avg_overall, 2)),
            "average_alignment_score": float(round(avg_alignment, 2)),
            "average_symmetry_score": float(round(avg_symmetry, 2)),
            "average_stability_score": float(round(avg_stability, 2)),
            "best_pose": best_pose,
            "weakest_pose": weakest_pose,
            "hold_times": holds,
            "most_frequent_errors": most_frequent_errors
        }

        # 3. Save Files
        # Save CSV Frame Logs
        csv_path = os.path.join(self.output_dir, f"session_logs_{timestamp_str}.csv")
        df.to_csv(csv_path, index=False)

        # Save Parquet Frame Logs for optimized storage
        try:
            parquet_path = os.path.join(self.output_dir, f"session_logs_{timestamp_str}.parquet")
            df.to_parquet(parquet_path, index=False)
        except Exception:
            parquet_path = None # Fallback silently if pyarrow/fastparquet is not fully set up

        # Save JSON Summary
        json_path = os.path.join(self.output_dir, f"session_summary_{timestamp_str}.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Append to historical progress file
        self._append_to_history(summary)

        return summary

    def _append_to_history(self, summary):
        """
        Appends summary to a global history log to track session-over-session trends.
        """
        history_path = os.path.join(self.output_dir, "history_trends.json")
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []
                
        history.append({
            "session_id": summary["session_id"],
            "date": time.strftime("%Y-%m-%d"),
            "practice_time_minutes": float(round(summary["total_practice_time_seconds"] / 60.0, 2)),
            "average_overall_score": summary["average_overall_score"],
            "best_pose": summary["best_pose"]
        })
        
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
