import os
import argparse
import cv2
import numpy as np
import time

from core.pose_detector import MediaPipePoseDetector, YOLOPoseDetector
from core.filter import SkeletalOneEuroFilter, SkeletalKalmanFilter, ExponentialMovingAverageFilter
from core.feature_extractor import FeatureExtractor
from core.classifier import YogaPoseClassifier
from core.similarity_engine import SimilarityEngine
from core.error_detector import ErrorDetector
from core.feedback_engine import FeedbackEngine
from core.scoring import ScoringEngine
from core.stability import StabilityAnalyzer
from core.symmetry import SymmetryAnalyzer
from core.state_machine import PoseStateMachine, PoseState
from core.analytics import SessionAnalytics
from visualization.visualizer import PoseVisualizer
from utils.logger import setup_logger

logger = setup_logger()

def parse_args():
    parser = argparse.ArgumentParser(description="AI Yoga Pose Detection & Posture Correction CLI")
    parser.add_argument("--source", type=str, default="0", 
                        help="Input source: '0'/'1' for webcam, path to video/image file, or RTSP URL")
    parser.add_argument("--backend", type=str, default="mediapipe", choices=["mediapipe", "yolo"],
                        help="Pose estimation model backend")
    parser.add_argument("--filter", type=str, default="one_euro", choices=["one_euro", "kalman", "ema", "none"],
                        help="Temporal coordinates smoothing filter")
    parser.add_argument("--config", type=str, default="config/poses_config.json",
                        help="Path to statistical poses config file")
    parser.add_argument("--weights", type=str, default="weights/pose_mlp.pth",
                        help="Path to trained MLP model weights")
    parser.add_argument("--out-dir", type=str, default="data/processed",
                        help="Folder to save session logs")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Initializing AI Yoga Instructor (Backend: {args.backend}, Filter: {args.filter})...")
    
    # 1. Initialize Pipeline Components
    # Backend detector
    if args.backend == "yolo":
        detector = YOLOPoseDetector()
    else:
        detector = MediaPipePoseDetector()
        
    # Filters
    if args.filter == "one_euro":
        coords_filter = SkeletalOneEuroFilter(min_cutoff=0.8, beta=0.03)
    elif args.filter == "kalman":
        coords_filter = SkeletalKalmanFilter()
    elif args.filter == "ema":
        coords_filter = ExponentialMovingAverageFilter(alpha=0.35)
    else:
        coords_filter = None
        
    classifier = YogaPoseClassifier(weights_path=args.weights, config_path=args.config)
    similarity_engine = SimilarityEngine(config_path=args.config)
    error_detector = ErrorDetector(config_path=args.config)
    feedback_engine = FeedbackEngine()
    scoring_engine = ScoringEngine()
    stability_analyzer = StabilityAnalyzer(window_size=30)
    symmetry_analyzer = SymmetryAnalyzer()
    state_machine = PoseStateMachine(alignment_threshold=0.80)
    analytics = SessionAnalytics(output_dir=args.out_dir)
    visualizer = PoseVisualizer()
    
    # 2. Setup Video Capture Source
    source = args.source
    # Check if source is integer (e.g. "0" for webcam)
    if source.isdigit():
        source = int(source)
        
    cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Optional: Try Full HD if your webcam supports it
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Camera Resolution:",
      cap.get(cv2.CAP_PROP_FRAME_WIDTH),
      "x",
      cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened():
        logger.error(f"Error: Could not open video source {args.source}")
        return

    # Check if input is a static image
    is_image = False
    if isinstance(source, str) and source.lower().endswith((".png", ".jpg", ".jpeg")):
        is_image = True

    logger.info("Yoga Instructor Pipeline active. Press 'q' to exit.")

    fps_timer = time.time()
    frame_count = 0
    fps = 0.0

    cv2.namedWindow("AI Yoga Instructor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Yoga Instructor", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_image:
                time.sleep(0.1)
                continue
            logger.info("End of video stream reached.")
            break
            
        frame_count += 1
        
        # 3. Pose Detection
        raw_landmarks, tracking_conf = detector.process_frame(frame)
        
        if raw_landmarks is not None:
            # 4. Temporal Coordinates Filtering
            if coords_filter is not None:
                smoothed_landmarks = coords_filter.filter_skeleton(raw_landmarks)
            else:
                smoothed_landmarks = raw_landmarks
                
            # 5. Feature Extraction
            skeleton = FeatureExtractor.normalize_skeleton(smoothed_landmarks, backend=args.backend)
            features = FeatureExtractor.extract_features(skeleton)
            feature_vector = FeatureExtractor.get_feature_vector(features)
            
            # 6. Pose Classification
            pose_name, class_conf, top_3 = classifier.classify(feature_vector, features)
            
            # 7. Posture Analysis
            # Cosine similarity and alignment z-scores
            sim_score, align_score, ref_conf, overall_sim = similarity_engine.compute_similarity(pose_name, features, skeleton)
            
            # Stability tracking
            stability_score, sway = stability_analyzer.update(skeleton)
            
            # Symmetry tracking
            symmetry_score, sym_breakdown = symmetry_analyzer.evaluate_symmetry(pose_name, features)
            
            # 8. State Machine & Timer Updates
            state, hold_duration, state_duration = state_machine.update(
                pose_name, align_score, person_detected=True, frame_confidence=tracking_conf
            )
            
            # 9. Error Detection & NL Feedback
            errors = error_detector.detect_errors(pose_name, features)
            feedback_tips, audio_text = feedback_engine.generate_feedback(errors)
            
            # 10. Posture Scoring & Explainable AI (XAI)
            overall_score, letter_grade, category, xai_deductions, breakdown = scoring_engine.compute_scores(
                errors, symmetry_score, stability_score, tracking_conf
            )
            
            # 11. Session Analytics Logging
            analytics.log_frame(pose_name, breakdown, errors, state.value)
            
            # 12. Visualization Renderings
            frame = visualizer.draw_pose(frame, skeleton, errors)
            frame = visualizer.draw_angles(frame, skeleton, features)
            frame = visualizer.draw_hud(frame, pose_name, state.value, hold_duration, overall_score, feedback_tips)
            
        else:
            # Handle No Person State
            state, hold_duration, _ = state_machine.update(
                None, 0.0, person_detected=False, frame_confidence=0.0
            )
            frame = visualizer.draw_hud(frame, None, state.value, 0.0, 0, [])
            stability_analyzer.reset()
            
        # Draw FPS counter
        if frame_count % 15 == 0:
            now = time.time()
            dt = now - fps_timer
            fps = 15.0 / dt if dt > 0 else 0.0
            fps_timer = now
            
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 90, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (46, 204, 113), 1, cv2.LINE_AA)

        # Show stream
        cv2.imshow("AI Yoga Instructor", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("Exiting pipeline loop...")
            break
            
        if is_image:
            # Keep showing the image until user quits
            cv2.waitKey(0)
            break

    # 13. Session Summary Logging
    cap.release()
    cv2.destroyAllWindows()
    
    session_holds = state_machine.get_session_analytics()
    summary = analytics.save_session(pose_holds=session_holds)
    
    if summary:
        print("\n" + "="*50)
        print("SESSION WORKOUT SUMMARY REPORT")
        print("="*50)
        print(f"Session Date:          {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Practice Time:   {summary['total_practice_time_seconds']:.1f} seconds")
        print(f"Average Practice Score: {summary['average_overall_score']}% (Grade: {scoring_engine.compute_scores([], 0, 0, 1.0)[1]})")
        print(f"Average Symmetry:      {summary['average_symmetry_score']}%")
        print(f"Average Stability:     {summary['average_stability_score']}%")
        print(f"Best Pose:             {summary['best_pose']}")
        print(f"Weakest Pose:          {summary['weakest_pose']}")
        
        print("\nPose Holds:")
        for pose, duration in summary["hold_times"].items():
            print(f" - {pose.replace('_', ' ').title()}: {duration:.1f}s")
            
        print("\nTop Posture Errors:")
        for idx, err in enumerate(summary["most_frequent_errors"]):
            print(f" {idx+1}. {err['joint'].replace('_', ' ').title()} ({err['count']} frames)")
        print("="*50)

if __name__ == "__main__":
    main()
