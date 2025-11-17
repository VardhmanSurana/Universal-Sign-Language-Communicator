
from utils import dataset_utils as du
if __name__ == "__main__":
    videos = du.load_dataset()
    reference_signs = du.load_reference_signs(videos)