import os
import cv2
import mediapipe as mp
import pickle
import numpy as np

DATA_VIDEOS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "videos")
REFERENCE_SIGNS_OUTPUT_PATH = "label_encoder.npy"
SEQUENCE_LENGTH = 50  # You might want to standardize the sequence length

def extract_landmarks_from_frame(results):
    lh_landmarks = []
    rh_landmarks = []
    if results.left_hand_landmarks:
        landmarks = results.left_hand_landmarks.landmark
        lh_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().tolist()
    if results.right_hand_landmarks:
        landmarks = results.right_hand_landmarks.landmark
        rh_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().tolist()
    return lh_landmarks, rh_landmarks

def create_reference_signs(videos_path, output_path, sequence_length):
    reference_signs_data = {}
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    for sign_name in os.listdir(videos_path):
        sign_dir_path = os.path.join(videos_path, sign_name)
        if os.path.isdir(sign_dir_path):
            print(f"Processing sign: {sign_name}")
            all_lh_sequences = []
            all_rh_sequences = []
            video_count = 0

            for video_file in [f for f in os.listdir(sign_dir_path) if f.endswith(".mp4")]:
                video_path = os.path.join(sign_dir_path, video_file)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error: Could not open video {video_path}")
                    continue

                lh_sequence = []
                rh_sequence = []
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = mp_hands.process(image)
                    lh_frame_landmarks, rh_frame_landmarks = extract_landmarks_from_frame(results)

                    if lh_frame_landmarks:
                        lh_sequence.append(lh_frame_landmarks)
                    else:
                        lh_sequence.append([0] * (21 * 3)) # Pad with zeros if no hand detected

                    if rh_frame_landmarks:
                        rh_sequence.append(rh_frame_landmarks)
                    else:
                        rh_sequence.append([0] * (21 * 3)) # Pad with zeros if no hand detected

                    frame_count += 1

                cap.release()

                # Standardize sequence length (you might need a more sophisticated approach)
                if len(lh_sequence) > 0 and len(rh_sequence) > 0:
                    if len(lh_sequence) >= sequence_length:
                        all_lh_sequences.append(lh_sequence[:sequence_length])
                        all_rh_sequences.append(rh_sequence[:sequence_length])
                        video_count += 1
                    elif len(lh_sequence) > 0:
                        # Pad with the last frame if shorter than desired length
                        padding_lh = [lh_sequence[-1]] * (sequence_length - len(lh_sequence))
                        padding_rh = [rh_sequence[-1]] * (sequence_length - len(rh_sequence))
                        all_lh_sequences.append(lh_sequence + padding_lh)
                        all_rh_sequences.append(rh_sequence + padding_rh)
                        video_count += 1

            if all_lh_sequences and all_rh_sequences:
                # For simplicity, we'll just take the landmark sequence from the first video for each sign.
                # You might want to average or use a more robust representation if you have multiple examples.
                reference_signs_data[sign_name] = {
                    'left_hand': all_lh_sequences[0],
                    'right_hand': all_rh_sequences[0]
                }
                print(f"  Processed {video_count} videos for {sign_name}. Using first sequence as reference.")
            else:
                print(f"  No valid landmark sequences found for {sign_name}.")

    mp_hands.close()

    with open(output_path, 'wb') as f:
        pickle.dump(reference_signs_data, f)

    print(f"\nReference sign data saved to: {output_path}")

if __name__ == "__main__":
    create_reference_signs(DATA_VIDEOS_PATH, REFERENCE_SIGNS_OUTPUT_PATH, SEQUENCE_LENGTH)