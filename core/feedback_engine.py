class FeedbackEngine:
    """
    Prioritizes posture correction guidelines for the user.
    Focuses on safety-critical alignment (spine, hips) before addressing limb adjustments (elbows, ankles).
    """
    def __init__(self):
        pass

    def _get_joint_priority(self, joint_name):
        """
        Assigns priority weights based on safety and biomechanical importance:
        Priority 1 (highest): Spine alignment, neck, back inclination
        Priority 2: Hip and shoulder alignment, knees
        Priority 3 (lowest): Elbows, ankles
        """
        name = joint_name.lower()
        if "spine" in name or "back" in name or "neck" in name:
            return 1
        elif "hip" in name or "shoulder" in name or "knee" in name:
            return 2
        return 3

    def generate_feedback(self, errors, max_tips=3):
        """
        Groups, prioritizes, and limits feedback strings to avoid cognitive overload.
        
        Returns:
            feedback_tips: List of dictionaries containing {joint, priority, tip, severity}
            audio_text: A concatenated string designed for text-to-speech feedback.
        """
        if not errors:
            return [], "Perfect alignment. Keep holding."

        prioritized_errors = []
        for err in errors:
            priority = self._get_joint_priority(err["joint"])
            prioritized_errors.append({
                "joint": err["joint"],
                "priority": priority,
                "tip": err["correction_tip"],
                "severity": err["severity"],
                "score_impact": err["weight"] * err["deviation"]
            })

        # Sort errors: Priority group (1 is highest), then score impact (descending)
        prioritized_errors.sort(key=lambda x: (x["priority"], -x["score_impact"]))

        # Select top N tips
        selected_tips = prioritized_errors[:max_tips]
        
        # Build text-to-speech feedback
        # Combine the top tips into a natural sentence, separating by slight pauses
        audio_phrases = []
        for item in selected_tips:
            audio_phrases.append(item["tip"])
            
        audio_text = ". ".join(audio_phrases) + "."
        
        return selected_tips, audio_text
