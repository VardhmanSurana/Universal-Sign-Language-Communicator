import os

import pandas as pd
from models import sign_model as sm
from utils import landmark_utils as lu


def load_dataset():
    videos = [
        file_name.replace(".mp4", "")
        for root, dirs, files in os.walk(os.path.join("data", "videos"))
        for file_name in files
        if file_name.endswith(".mp4")
    ]
    dataset = [
    file_name.replace(".csv", "")
    for root, dirs, files in os.walk(os.path.join("data", "dataset"))
    for file_name in files
    if file_name.endswith(".csv")
]


    # Create the dataset from the reference videos
    videos_not_in_dataset = list(set(videos).difference(set(dataset)))
    n = len(videos_not_in_dataset)
    if n > 0:

        for idx in range(n):
            print(f"Saving {videos_not_in_dataset[idx]}...")
            lu.save_landmarks_from_video(videos_not_in_dataset[idx])

    return videos


def load_reference_signs(videos):
    reference_signs = {"name": [], "sign_model": [], "distance": []}
    for video_name in videos:
        sign_name = video_name.split("-")[0]
        path = os.path.join("data", "dataset", sign_name, video_name)

        left_hand_list = lu.load_array(os.path.join(path, f"lh_{video_name}.pickle"))
        right_hand_list = lu.load_array(os.path.join(path, f"rh_{video_name}.pickle"))

        reference_signs["name"].append(sign_name)
        reference_signs["sign_model"].append(sm.SignModel(left_hand_list, right_hand_list))
        reference_signs["distance"].append(0)

    reference_signs = pd.DataFrame(reference_signs, dtype=object)
    return reference_signs