import pandas as pd
from fastdtw import fastdtw
import numpy as np
from models import sign_model as sm


def dtw_distances(recorded_sign: sm.SignModel, reference_signs: pd.DataFrame):
    rec_left_hand = recorded_sign.lh_embedding
    rec_right_hand = recorded_sign.rh_embedding

    for idx, row in reference_signs.iterrows():
        ref_sign_name, ref_sign_model, _ = row

        if (recorded_sign.has_left_hand == ref_sign_model.has_left_hand) and (
            recorded_sign.has_right_hand == ref_sign_model.has_right_hand
        ):
            ref_left_hand = ref_sign_model.lh_embedding
            ref_right_hand = ref_sign_model.rh_embedding

            if recorded_sign.has_left_hand:
                row["distance"] += list(fastdtw(rec_left_hand, ref_left_hand))[0]
            if recorded_sign.has_right_hand:
                row["distance"] += list(fastdtw(rec_right_hand, ref_right_hand))[0]
        else:
            row["distance"] = np.inf
    return reference_signs.sort_values(by=["distance"])
