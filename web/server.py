import os
import sys
import json
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Ensure root folder is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose_detector import MediaPipePoseDetector
from core.filter import SkeletalOneEuroFilter
from core.feature_extractor import FeatureExtractor
from core.classifier import YogaPoseClassifier
from core.similarity_engine import SimilarityEngine
from core.error_detector import ErrorDetector
from core.feedback_engine import FeedbackEngine
from core.scoring import ScoringEngine
from core.stability import StabilityAnalyzer
from core.symmetry import SymmetryAnalyzer
from core.state_machine import PoseStateMachine
from core.analytics import SessionAnalytics

app = FastAPI(title="AI Yoga Pose Detection & Posture Correction System")

# Ensure static folder exists
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def get_index():
    """
    Renders main dashboard page.
    """
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>Loading AI Yoga Instructor UI... Please create index.html in web/static/</h1>", status_code=200)

@app.get("/api/poses")
def get_poses():
    """
    Returns lists of available poses and details for the frontend dropdown list.
    """
    config_path = "config/poses_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {"poses": {}}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 1. Initialize session-level AI components for the WebSocket client
    detector = MediaPipePoseDetector(model_complexity=1)
    coords_filter = SkeletalOneEuroFilter(min_cutoff=0.8, beta=0.03)
    classifier = YogaPoseClassifier()
    similarity_engine = SimilarityEngine()
    error_detector = ErrorDetector()
    feedback_engine = FeedbackEngine()
    scoring_engine = ScoringEngine()
    stability_analyzer = StabilityAnalyzer(window_size=30)
    symmetry_analyzer = SymmetryAnalyzer()
    state_machine = PoseStateMachine(alignment_threshold=0.80)
    analytics = SessionAnalytics()
    
    selected_target_pose = None

    try:
        while True:
            # 2. Receive WebSocket messages
            message = await websocket.receive()
            
            # Check text commands (e.g. changing the targeted pose)
            if "text" in message:
                try:
                    cmd_data = json.loads(message["text"])
                    if cmd_data.get("action") == "set_pose":
                        selected_target_pose = cmd_data.get("pose_name")
                        state_machine.current_pose = selected_target_pose
                        state_machine.reset_timer()
                        stability_analyzer.reset()
                    elif cmd_data.get("action") == "save_session":
                        holds = state_machine.get_session_analytics()
                        summary = analytics.save_session(pose_holds=holds)
                        await websocket.send_text(json.dumps({
                            "type": "session_summary",
                            "summary": summary
                        }))
                        analytics.reset()
                except Exception as e:
                    pass
                continue
                
            # Check frame byte payloads
            if "bytes" not in message:
                continue
                
            frame_bytes = message["bytes"]
            
            # 3. Decode JPEG frame to OpenCV image
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            # 4. Execute AI Pipeline
            raw_landmarks, tracking_conf = detector.process_frame(frame)
            
            response_data = {
                "type": "pipeline_update",
                "person_detected": False,
                "pose_name": None,
                "state": "No Person Detected",
                "hold_duration": 0.0,
                "overall_score": 0,
                "breakdown": {"alignment": 0.0, "symmetry": 0.0, "stability": 0.0, "confidence": 0.0},
                "skeleton": {},
                "angles": {},
                "errors": [],
                "deductions": {},
                "top_predictions": []
            }
            
            if raw_landmarks is not None:
                # Smooth coordinates
                smoothed_landmarks = coords_filter.filter_skeleton(raw_landmarks)
                
                # Extract features
                skeleton = FeatureExtractor.normalize_skeleton(smoothed_landmarks, backend="mediapipe")
                features = FeatureExtractor.extract_features(skeleton)
                feature_vector = FeatureExtractor.get_feature_vector(features)
                
                # Classify pose
                classified_pose, class_conf, top_3 = classifier.classify(feature_vector, features)
                
                # Use user's selected pose if specified, otherwise fall back to classified pose
                active_pose = selected_target_pose if selected_target_pose else classified_pose
                
                # Run evaluation engines
                sim_score, align_score, ref_conf, overall_sim = similarity_engine.compute_similarity(active_pose, features, skeleton)
                stability_score, sway = stability_analyzer.update(skeleton)
                symmetry_score, sym_breakdown = symmetry_analyzer.evaluate_symmetry(active_pose, features)
                
                # State Machine
                state, hold_duration, state_duration = state_machine.update(
                    active_pose, align_score, person_detected=True, frame_confidence=tracking_conf
                )
                
                # Errors & feedback
                errors = error_detector.detect_errors(active_pose, features)
                feedback_tips, audio_text = feedback_engine.generate_feedback(errors)
                
                # Scores & XAI
                overall_score, letter_grade, category, xai_deductions, breakdown = scoring_engine.compute_scores(
                    errors, symmetry_score, stability_score, tracking_conf
                )
                
                # Log frame to session
                analytics.log_frame(active_pose, breakdown, errors, state.value)
                
                # Prepare JSON response payload
                # Serialize landmarks as standard 3D lists to draw on canvas
                serialized_skeleton = {name: [float(pt[0]), float(pt[1]), float(pt[2]), float(pt[3])] for name, pt in skeleton.items()}
                
                # Filter key angles to display
                display_angles = {
                    "left_elbow": float(features.get("left_elbow", 180.0)),
                    "right_elbow": float(features.get("right_elbow", 180.0)),
                    "left_knee": float(features.get("left_knee", 180.0)),
                    "right_knee": float(features.get("right_knee", 180.0)),
                    "left_hip": float(features.get("left_hip", 180.0)),
                    "right_hip": float(features.get("right_hip", 180.0)),
                    "spine_alignment": float(features.get("spine_alignment", 0.0))
                }
                
                # Format error tips
                formatted_errors = [
                    {
                        "joint": err["joint"].replace("_", " ").title(),
                        "deviation": float(err["deviation"]),
                        "severity": err["severity"],
                        "tip": err["correction_tip"]
                    }
                    for err in errors
                ]

                response_data.update({
                    "person_detected": True,
                    "pose_name": active_pose,
                    "state": state.value,
                    "hold_duration": float(hold_duration),
                    "overall_score": int(overall_score),
                    "letter_grade": letter_grade,
                    "category": category,
                    "breakdown": breakdown,
                    "skeleton": serialized_skeleton,
                    "angles": display_angles,
                    "errors": formatted_errors,
                    "deductions": xai_deductions,
                    "top_predictions": top_3,
                    "tts_feedback": audio_text
                })
            else:
                # Handle empty frame states
                state, _, _ = state_machine.update(
                    None, 0.0, person_detected=False, frame_confidence=0.0
                )
                response_data["state"] = state.value
                stability_analyzer.reset()
                
            # Send payload back
            await websocket.send_text(json.dumps(response_data))
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
