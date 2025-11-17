class SignRecorder:
    def __init__(self):
        pass

    def get_keypoints(self, results):
        """
        Extracts keypoints from both hands.
        Returns a flattened list of coordinates.
        """
        keypoints = []

        def extract_hand_landmarks(hand_landmarks):
            if hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            else:
                # Fill with zeros if hand is not detected
                keypoints.extend([0.0] * 21 * 3)

        extract_hand_landmarks(results.left_hand_landmarks)
        extract_hand_landmarks(results.right_hand_landmarks)

        return keypoints if any(keypoints) else None
