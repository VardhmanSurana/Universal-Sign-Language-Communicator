import os
import cv2
import csv
import logging
import traceback
import numpy as np
import mediapipe as mp
from utils import dataset_utils as du
from utils import mediapipe_utils as mu
from scipy.spatial.distance import euclidean

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_distances(landmarks):
    """Calculates Euclidean distances between all pairs of keypoints."""
    distances = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            dist = euclidean(landmarks[i], landmarks[j])
            distances.append(dist)
    return distances

def calculate_angles(landmarks):
    """Calculates angles between joints formed by three consecutive keypoints."""
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
    return angles

def normalize_landmarks(landmarks):
    """Normalizes landmarks relative to the wrist joint (index 0)."""
    if not landmarks:
        return []
    wrist = np.array(landmarks[0])
    normalized = (np.array(landmarks) - wrist).flatten().tolist()
    return normalized

def save_landmarks_from_video(video_name):
    """
    Extracts left hand, and right hand features (landmarks, distances, angles, normalized coordinates)
    from a video and saves them to a CSV file, skipping frames where both hands have all zero
    keypoint values.
    """
    sign_name = video_name.split("-")[0]
    video_path = os.path.join("data", "videos", sign_name, video_name + ".mp4")

    logging.info(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Define the output CSV path
    dataset_path = os.path.join("data", "dataset", sign_name)
    os.makedirs(dataset_path, exist_ok=True)  # Create directory if not exists
    csv_file_path = os.path.join(dataset_path, f"{video_name}.csv")

    # Define CSV header
    lh_landmark_headers = [f"lh_landmark_{i}" for i in range(21 * 3)]
    rh_landmark_headers = [f"rh_landmark_{i}" for i in range(21 * 3)]
    lh_distance_headers = [f"lh_dist_{i}" for i in range(21 * 20 // 2)] # nC2 combinations
    rh_distance_headers = [f"rh_dist_{i}" for i in range(21 * 20 // 2)]
    lh_angle_headers = [f"lh_angle_{i}" for i in range(21 - 2)]
    rh_angle_headers = [f"rh_angle_{i}" for i in range(21 - 2)]
    lh_normalized_headers = [f"lh_norm_{i}" for i in range(21 * 3)]
    rh_normalized_headers = [f"rh_norm_{i}" for i in range(21 * 3)]

    headers = ["frame"] + \
              lh_landmark_headers + rh_landmark_headers + \
              lh_distance_headers + rh_distance_headers + \
              lh_angle_headers + rh_angle_headers + \
              lh_normalized_headers + rh_normalized_headers

    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write header row

        with mp.solutions.holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.25) as holistic:
            frame_number = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    image, results = mu.mediapipe_detection(frame, holistic)
                    lh_landmarks, rh_landmarks = extract_all_features(results)

                    # Check if both left and right hand keypoints are all zeros
                    if not (all(x == 0 for x in lh_landmarks[:63]) and all(x == 0 for x in rh_landmarks[:63])):
                        # Combine all data into a single row
                        row_data = [frame_number] + lh_landmarks + rh_landmarks
                        writer.writerow(row_data)
                        frame_number += 1
                    else:
                        logging.info(f"Skipping frame {frame_number} in {video_name} as both hands have no keypoints.")

                except Exception as e:
                    logging.error(f"Error processing frame in {video_name}: {e}")
                    traceback.print_exc()

    cap.release()
    logging.info(f"Saved dataset to {csv_file_path}")

def extract_all_features(results):
    """
    Extracts landmarks, distances, angles, and normalized coordinates for both hands.
    """
    lh_landmarks_flat = np.zeros(21 * 3).tolist()
    rh_landmarks_flat = np.zeros(21 * 3).tolist()
    lh_distances = []
    rh_distances = []
    lh_angles = []
    rh_angles = []
    lh_normalized = []
    rh_normalized = []

    if results.left_hand_landmarks:
        lh_landmarks = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        lh_landmarks_flat = np.array(lh_landmarks).flatten().tolist()
        lh_distances = calculate_distances(lh_landmarks)
        lh_angles = calculate_angles(lh_landmarks)
        lh_normalized = normalize_landmarks(lh_landmarks)

    if results.right_hand_landmarks:
        rh_landmarks = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        rh_landmarks_flat = np.array(rh_landmarks).flatten().tolist()
        rh_distances = calculate_distances(rh_landmarks)
        rh_angles = calculate_angles(rh_landmarks)
        rh_normalized = normalize_landmarks(rh_landmarks)

    # Pad with zeros to maintain consistent feature vector length
    lh_all = lh_landmarks_flat + lh_distances + lh_angles + lh_normalized
    rh_all = rh_landmarks_flat + rh_distances + rh_angles + rh_normalized

    # Ensure consistent length (padding with zeros if necessary)
    expected_lh_len = (21 * 3) + (21 * 20 // 2) + (21 - 2) + (21 * 3)
    expected_rh_len = (21 * 3) + (21 * 20 // 2) + (21 - 2) + (21 * 3)

    lh_padded = lh_all + [0.0] * (expected_lh_len - len(lh_all))
    rh_padded = rh_all + [0.0] * (expected_rh_len - len(rh_all))

    return lh_padded, rh_padded

def extract_landmarks(results):
    """
    Extracts left hand, and right hand landmarks from MediaPipe results (legacy function).
    """
    left_hand = np.zeros(63).tolist()
    right_hand = np.zeros(63).tolist()

    if results.left_hand_landmarks:
        left_hand = landmark_to_array(results.left_hand_landmarks, 63)

    if results.right_hand_landmarks:
        right_hand = landmark_to_array(results.right_hand_landmarks, 63)

    return left_hand, right_hand

def landmark_to_array(mp_landmark_list, expected_length):
    """
    Converts MediaPipe landmarks into a flattened NumPy array with NaN handling (legacy function).
    """
    keypoints = [[lm.x, lm.y, lm.z] for lm in mp_landmark_list.landmark]
    flattened = np.array(keypoints).flatten()

    # Ensure the array has the correct size
    if len(flattened) < expected_length:
        flattened = np.concatenate([flattened, np.zeros(expected_length - len(flattened))])

    return flattened.tolist()

if __name__ == "__main__":
    try:
        videos = du.load_dataset()
        if not videos:
            logging.error("No videos found. Check the 'data/videos' directory.")
            exit()

        for video in videos:
            save_landmarks_from_video(video)

        logging.info("Landmark and feature extraction complete.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()