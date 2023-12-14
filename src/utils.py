import pandas as pd
from typing import List
from copy import copy
import numpy as np
from typing import Union
from enum import Enum

from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def slice_by(df: pd.DataFrame, column_name_to_slice_by: str) -> List[pd.DataFrame]:
    """
    :param df: dataframe with multiple time series
    :param column_name_to_slice_by: str, column name to identify unique time series
    :return: list of dataframes.
    NOTE: since this function uses pandas groupby method the order of the list the order won't be preserved
    """
    ret = []
    for _, group in df.groupby(column_name_to_slice_by):
        ret.append(group)
    return ret


class Methods(str, Enum):
    min_max = "min_max"
    standard = "standard"


def select_scaler(method: str):
    if method == Methods.standard:
        scaler = StandardScaler()
    elif method == Methods.min_max:
        scaler = MinMaxScaler()
    else:
        raise RuntimeError("Something went wrong, no scaling method chosen")
    return scaler


def within_subject_functional_normalization(x: Union[np.ndarray, pd.DataFrame],
                                            subject_ids: np.ndarray,
                                            method: str) -> Union[np.ndarray, pd.DataFrame]:
    """
    Normalization function whereby each subject is normalized separately

    :param x: matrix with shape (observations, features)
    :param subject_ids: np Array with subject_ids
    :param method: normalization method
    :return: normalized matrix
    """

    # make a copy of x to not modify input variable
    ret_x = copy(x)

    # iterate over all subject ids
    for subject_id in np.unique(subject_ids):

        # select normalization method and create a scaler for the specific subject
        scaler = select_scaler(method)

        # normalize the rows corresponding for the specific subject
        rows = np.where(subject_ids == subject_id)[0]

        if isinstance(x, pd.DataFrame):
            ret_x.iloc[rows] = scaler.fit_transform(x.iloc[rows])
        elif isinstance(x, np.ndarray):
            ret_x[rows] = scaler.fit_transform(x[rows])
        else:
            raise ValueError("input array x need to be either a pd.Dataframe or np.ndarray")

    return ret_x


def evaluate_scores(x, y, clf, scoring_method):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

    splits = skf.split(x, y)

    # get scores
    scores = cross_validate(X=x, y=y,
                            estimator=clf,
                            scoring=[scoring_method],
                            cv=splits,
                            n_jobs=-1,
                            return_train_score=True
                            )

    print('\nprinting {} measures'.format(scoring_method))
    print('avg (train):', np.mean(scores['train_{}'.format(scoring_method)]))
    print('std (train):', np.std(scores['train_{}'.format(scoring_method)]))
    print('avg (validation):', np.mean(scores['test_{}'.format(scoring_method)]))
    print('std (validation):', np.std(scores['test_{}'.format(scoring_method)]))


def get_splits(x, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    return skf.split(x, y)
