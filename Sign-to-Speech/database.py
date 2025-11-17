import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from utils import dataset_utils as du
from utils import landmark_utils as lu

def load_reference_signs_from_database():
    """Loads the reference signs DataFrame."""
    videos = du.load_dataset()
    reference_signs = du.load_reference_signs(videos)
    return reference_signs

def create_new_landmark_data():
    """Processes new videos to create landmark data."""
    videos = du.load_dataset() # This will handle creating new landmark data if needed
    print("New landmark data processing completed (if any).")

if __name__ == '__main__':
    print("Running database.py to load/create dataset...")
    create_new_landmark_data()
    reference_signs_df = load_reference_signs_from_database()
    print("\nReference Signs DataFrame:")
    print(reference_signs_df.head())
    print("\nDatabase loading/creation complete.")