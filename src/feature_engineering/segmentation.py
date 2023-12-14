import numpy as np


def segment_based_on_silence(f0_vector: np.ndarray) -> (list, list):
    """
    :param f0_vector: vector with F0 values, F0=0 means silence, F!=0 means voice
    :return: list of continuously silent segments, and list continuously voiced segments
    """
    silence_vec = f0_vector == 0
    voice_vec = f0_vector != 0
    silent_segments = find_continuous_segments(silence_vec)
    voiced_segments = find_continuous_segments(voice_vec)

    return silent_segments, voiced_segments


def find_continuous_segments(boolean_vector: np.ndarray) -> list:
    """
    :param boolean_vector: vector with true or false values
    :return: list of segments that are continuously true
    """
    # we don't consider continuous segments that are no greater than threshold
    threshold = 2

    continuous_segments = []
    n_segments = 0
    for val in boolean_vector:
        if val:
            n_segments += 1
        else:
            if n_segments > threshold:
                continuous_segments.append(n_segments)
                n_segments = 0

    if n_segments > threshold:
        continuous_segments.append(n_segments)
    return continuous_segments
