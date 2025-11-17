import cv2
import numpy as np
import mediapipe as mp
import time
from scipy.spatial.distance import euclidean
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic

# Load the trained model
try:
    model = load_model("sign_language_model.keras")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load label encoder
try:
    label_encoder = np.load("label_encoder.npy", allow_pickle=True)
    print("Label encoder loaded successfully.")
except FileNotFoundError:
    print("Error: label_encoder.npy not found. Ensure it's in the correct directory.")
    label_encoder = None

def calculate_distances(landmarks):
    """Calculates Euclidean distances between all pairs of keypoints."""
    print("Calculating distances...")
    distances = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            dist = euclidean(landmarks[i], landmarks[j])
            distances.append(dist)
    print("Distances calculated:")
    return distances

def calculate_angles(landmarks):
    """Calculates angles between joints formed by three consecutive keypoints."""
    print("Calculating angles...")
    angles = []
    for i in range(len(landmarks) - 2):
        p1 = np.array(landmarks[i])
        p2 = np.array(landmarks[i+1])
        p3 = np.array(landmarks[i+2])

        v1 = p1 - p2
        v2 = p3 - p2

        # Normalize vectors to get the cosine of the angle
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm > 0 and v2_norm > 0:
            cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
            # Ensure cos_theta is within the valid range [-1, 1] due to potential floating-point errors
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.arccos(cos_theta)
            angles.append(np.degrees(angle))
        else:
            angles.append(0.0)  # Handle cases where a vector has zero length
    print("Angles calculated:")
    return angles

def normalize_landmarks(landmarks):
    """Normalizes landmarks relative to the wrist joint (index 0)."""
    print("Normalizing landmarks...")
    if not landmarks:
        return []
    wrist = np.array(landmarks[0])
    normalized = (np.array(landmarks) - wrist).flatten().tolist()
    print("Landmarks normalized:")
    return normalized

def extract_all_features(results, keypoint_counter):
    """
    Extracts landmarks, distances, angles, and normalized coordinates for both hands,
    ensuring a consistent feature vector length.
    """
    print("Extracting features...")
    lh_landmark_count = 21 * 3  # x, y, z for 21 landmarks
    rh_landmark_count = 21 * 3
    num_distances = 21 * (21 - 1) // 2
    num_angles = 21 - 2
    num_normalized = 21 * 3

    lh_expected_len = lh_landmark_count + num_distances + num_angles + num_normalized
    rh_expected_len = rh_landmark_count + num_distances + num_angles + num_normalized
    total_expected_len = lh_expected_len + rh_expected_len

    lh_features = np.zeros(lh_expected_len).tolist()
    rh_features = np.zeros(rh_expected_len).tolist()

    left_hand_landmarks_present = False
    right_hand_landmarks_present = False

    if results.left_hand_landmarks:
        left_hand_landmarks_present = True
        lh_landmarks = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        lh_landmarks_flat = np.array(lh_landmarks).flatten().tolist()
        lh_distances = calculate_distances(lh_landmarks)
        lh_angles = calculate_angles(lh_landmarks)
        lh_normalized = normalize_landmarks(lh_landmarks)
        lh_current_features = lh_landmarks_flat + lh_distances + lh_angles + lh_normalized
        lh_features[:len(lh_current_features)] = lh_current_features

    if results.right_hand_landmarks:
        right_hand_landmarks_present = True
        rh_landmarks = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        rh_landmarks_flat = np.array(rh_landmarks).flatten().tolist()
        rh_distances = calculate_distances(rh_landmarks)
        rh_angles = calculate_angles(rh_landmarks)
        rh_normalized = normalize_landmarks(rh_landmarks)
        rh_current_features = rh_landmarks_flat + rh_distances + rh_angles + rh_normalized
        rh_features[:len(rh_current_features)] = rh_current_features

    if left_hand_landmarks_present or right_hand_landmarks_present:
        keypoint_counter[0] += 1
        print("Keypoints detected and counter incremented.")
    else:
        print("No hand keypoints detected in this frame.")

    # Combine left and right hand features
    all_features = lh_features + rh_features
    print(f"Length of extracted features: {len(all_features)}")  # Debugging
    print("All features extracted:")
    return np.array(all_features), keypoint_counter

def predict_sign_with_features():
    print("Running predict_sign_with_features...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    keypoint_extracted_count = [0]  # Use a list to pass by reference

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        print("Holistic model loaded, starting loop...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            features, keypoint_extracted_count = extract_all_features(results, keypoint_extracted_count)

            cv2.putText(image, f"Keypoints Extracted: {keypoint_extracted_count[0]}", (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if features.size > 0 and model is not None and label_encoder is not None:
                features_reshaped = features.reshape(1, -1)
                prediction = model.predict(features_reshaped)
                predicted_class = np.argmax(prediction)
                predicted_sign = label_encoder[predicted_class]
                confidence = np.max(prediction)

                cv2.putText(image, f"Sign: {predicted_sign} ({confidence:.2f})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            elif model is None or label_encoder is None:
                cv2.putText(image, "Model or Label Encoder Missing", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(image, "No Features Detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.imshow("Sign Language Recognition", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Total frames where keypoints were extracted: {keypoint_extracted_count[0]}")

if __name__ == "__main__":
    predict_sign_with_features()