import numpy as np

class ScoringEngine:
    """
    Computes a comprehensive, multi-dimensional posture score (0-100) and maps it to
    a letter grade and classification. Implements Explainable AI (XAI) score deductions.
    """
    def __init__(self):
        pass

    def compute_scores(self, errors, symmetry_score, stability_score, tracking_confidence):
        """
        Calculates sub-scores and compiles them into a unified evaluation.
        
        Returns:
            overall_score: int (0 to 100)
            letter_grade: str (A, B, C, D, F)
            category: str (Excellent, Very Good, Good, Needs Improvement, Poor)
            xai_deductions: dict (maps joints to score deductions)
            breakdown: dict of all sub-scores
        """
        # 1. Compute Alignment Deductions & XAI Attribution
        xai_deductions = {}
        total_alignment_deduction = 0.0
        
        for err in errors:
            joint = err["joint"]
            deviation = err["deviation"]
            target_mean = err["target_mean"]
            # Get weights and configurations
            weight = err.get("weight", 1.0)
            severity = err.get("severity", "Low")
            
            # Penalize deviation using an exponential scaling relative to severity
            if severity == "Low":
                factor = 0.3
            elif severity == "Medium":
                factor = 0.8
            else:
                factor = 1.8
                
            deduction = min(25.0, deviation * weight * factor * 0.15)
            deduction_round = int(round(deduction))
            
            if deduction_round > 0:
                display_name = joint.replace("_", " ").title()
                xai_deductions[display_name] = -deduction_round
                total_alignment_deduction += deduction

        alignment_score = max(0.0, 100.0 - total_alignment_deduction)

        # 2. Weighted Overall Score Composition
        # Weighting: Alignment (60%), Stability (20%), Symmetry (20%)
        # Penalize if tracking confidence is extremely low
        overall = (
            0.60 * alignment_score +
            0.20 * (symmetry_score * 100.0) +
            0.20 * (stability_score * 100.0)
        )
        
        # Scaling based on tracking confidence (occlusions reduce confidence)
        if tracking_confidence < 0.6:
            overall = overall * (tracking_confidence / 0.6)
            
        overall_score = int(np.clip(round(overall), 0, 100))

        # 3. Letter Grade & Category Mapping
        if overall_score >= 90:
            letter_grade = "A"
            category = "Excellent"
        elif overall_score >= 80:
            letter_grade = "B"
            category = "Very Good"
        elif overall_score >= 70:
            letter_grade = "C"
            category = "Good"
        elif overall_score >= 50:
            letter_grade = "D"
            category = "Needs Improvement"
        else:
            letter_grade = "F"
            category = "Poor"

        breakdown = {
            "alignment": float(round(alignment_score, 1)),
            "symmetry": float(round(symmetry_score * 100.0, 1)),
            "stability": float(round(stability_score * 100.0, 1)),
            "confidence": float(round(tracking_confidence * 100.0, 1))
        }

        return overall_score, letter_grade, category, xai_deductions, breakdown
