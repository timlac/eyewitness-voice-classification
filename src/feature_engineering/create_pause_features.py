import numpy as np
import pandas as pd

from src.utils import slice_by
from segmentation import segment_based_on_silence

master_low_level_path = 'data/master_low_level.csv'
master_functionals_path = 'data/master_functionals.csv'
master_functionals_new_features_path = 'data/master_functionals_new_features.csv'


def get_number_of_pauses_above_length(pause_lengths, threshold):
    ret = 0
    for pause_length in pause_lengths:
        if pause_length > threshold:
            ret += 1
    return ret


# Read functionals, initialize new features
df_funcs = pd.read_csv(master_functionals_path)
df_funcs["CumPauseLength"] = float(0)

df_funcs["MeanPauseSegmentLength"] = float(0)

df_funcs["PausesAbove0.5Sec"] = float(0)
df_funcs["PausesAbove1Sec"] = float(0)
df_funcs["PausesAbove1.5Sec"] = float(0)
df_funcs["PausesAbove2Sec"] = float(0)

# read llds, slice by filename
df_llds = pd.read_csv(master_low_level_path)
slices = slice_by(df_llds, "filename")

for idx, df_lld in enumerate(slices):

    # obtain filename for current slice
    filename = df_lld["filename"].iloc[0]

    # obtain corresponding row in functionals file
    df_func = df_funcs.loc[df_funcs['filename'] == filename]

    total_duration = pd.to_timedelta(df_lld["end"].max()).total_seconds()
    total_segments = len(df_lld)

    silent_segment_lengths, voiced_segment_lengths = segment_based_on_silence(
        df_lld['F0semitoneFrom27.5Hz_sma3nz'].values)

    if len(silent_segment_lengths) > 2:
        # finding pauses is as easy as simply removing the first and last silent segments.
        pauses_segment_lengths = silent_segment_lengths[1:-1]
        cum_pause_length = sum(pauses_segment_lengths) / 100
        mean_length_pauses = np.asarray(pauses_segment_lengths).mean() / 100

        condition = df_funcs['filename'] == filename
        df_funcs.loc[condition, 'CumPauseLength'] = cum_pause_length
        df_funcs.loc[condition, 'MeanPauseSegmentLength'] = mean_length_pauses

        df_funcs.loc[condition, 'PausesAbove0.5Sec'] = get_number_of_pauses_above_length(pauses_segment_lengths, 50)
        df_funcs.loc[condition, 'PausesAbove1Sec'] = get_number_of_pauses_above_length(pauses_segment_lengths, 100)
        df_funcs.loc[condition, 'PausesAbove1.5Sec'] = get_number_of_pauses_above_length(pauses_segment_lengths, 150)
        df_funcs.loc[condition, 'PausesAbove2Sec'] = get_number_of_pauses_above_length(pauses_segment_lengths, 200)

df_funcs.to_csv(master_functionals_new_features_path)
