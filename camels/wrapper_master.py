from typing import Tuple, Any
from scipy.stats import skew, kurtosis, entropy
import pandas as pd
import numpy as np
from collections import Counter

from camels.client.client_database_identifier import Algorithm, Metadata, algorithm_library_map

from camels.wrapper_lenskit import fit_and_predict as lenskit_fp
from camels.wrapper_lenskit import train_best_model as lenskit_tb
from camels.wrapper_surprise import fit_and_predict as surprise_fp
from camels.wrapper_surprise import train_best_model as surprise_tb
from camels.wrapper_myfm import fit_and_predict as myfm_fp
from camels.wrapper_myfm import train_best_model as myfm_tb
from camels.wrapper_sklearn import fit_and_predict as sklearn_fp
from camels.wrapper_sklearn import train_best_model as sklearn_tb
from camels.wrapper_xgboost import fit_and_predict as xgboost_fp
from camels.wrapper_xgboost import train_best_model as xgboost_tb


def calculate_meta_features(data: pd.DataFrame) -> dict:
    # some data stats
    min_rating = data["rating"].min()
    max_rating = data["rating"].max()
    mean_rating = data["rating"].mean()
    median_rating = data["rating"].median()
    mode_rating = data["rating"].mode()[0]
    user_counter = Counter(data["user"])
    item_counter = Counter(data["item"])
    num_users = data["user"].unique().size
    num_items = data["item"].unique().size
    num_instances = len(data)
    density = (num_instances * 100) / (num_users * num_items)
    unique_ratings, rating_counts = np.unique(data["rating"], return_counts=True)
    increments = []
    for idx in range(len(unique_ratings) - 1):
        increments.append(abs(unique_ratings[idx + 1] - unique_ratings[idx]))

    # create meta data object
    meta_data = {
        Metadata.NumUsers.name: num_users,
        Metadata.NumItems.name: num_items,
        Metadata.NumInstances.name: num_instances,
        Metadata.Sparsity.name: 100 - density,
        Metadata.Density.name: density,
        Metadata.MinRating.name: min_rating.tolist(),
        Metadata.MaxRating.name: max_rating.tolist(),
        Metadata.MeanRating.name: mean_rating.tolist(),
        Metadata.MedianRating.name: median_rating.tolist(),
        Metadata.ModeRating.name: mode_rating.tolist(),
        Metadata.NormalizedMeanRating.name: ((mean_rating - min_rating) / (max_rating - min_rating)).tolist(),
        Metadata.NormalizedMedianRating.name: ((median_rating - min_rating) / (max_rating - min_rating)).tolist(),
        Metadata.NormalizedModeRating.name: ((mode_rating - min_rating) / (max_rating - min_rating)).tolist(),
        Metadata.UserItemRatio.name: num_users / num_items,
        Metadata.ItemUserRatio.name: num_items / num_users,
        Metadata.HighestNumRatingBySingleUser.name: user_counter.most_common()[0][1],
        Metadata.LowestNumRatingBySingleUser.name: user_counter.most_common()[-1][1],
        Metadata.HighestNumRatingOnSingleItem.name: item_counter.most_common()[0][1],
        Metadata.LowestNumRatingOnSingleItem.name: item_counter.most_common()[-1][1],
        Metadata.MeanNumRatingsByUser.name: num_instances / num_users,
        Metadata.MeanNumRatingsOnItem.name: num_instances / num_items,
        Metadata.RatingSkew.name: skew(data["rating"]),
        Metadata.RatingKurtosis.name: kurtosis(data["rating"]),
        Metadata.RatingsStandardVariation.name: data["rating"].std().tolist(),
        Metadata.RatingVariance.name: data["rating"].var().tolist(),
        Metadata.RatingEntropy.name: (entropy(rating_counts) / np.log(len(rating_counts))).tolist(),
        Metadata.NumPossibleRatings.name: len(data["rating"].unique()),
        Metadata.RatingAverageIncrement.name: sum(increments) / len(increments)
    }

    return meta_data


def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame, algorithm: Algorithm) -> Tuple[pd.DataFrame, float, float]:
    if algorithm in algorithm_library_map["lenskit_algorithms"]:
        return lenskit_fp(train, test, algorithm)
    elif algorithm in algorithm_library_map["surprise_algorithms"]:
        return surprise_fp(train, test, algorithm)
    elif algorithm in algorithm_library_map["myfm_algorithms"]:
        return myfm_fp(train, test, algorithm)
    elif algorithm in algorithm_library_map["sklearn_algorithms"]:
        return sklearn_fp(train, test, algorithm)
    elif algorithm in algorithm_library_map["xgboost_algorithms"]:
        return xgboost_fp(train, test, algorithm)
    else:
        print(f"Exception: The algorithm {Algorithm.name} is unsupported.")
        raise Exception


def train_best_model(data: pd.DataFrame, algorithm: Algorithm) -> Any:
    if algorithm in algorithm_library_map["lenskit_algorithms"]:
        return lenskit_tb(data, algorithm)
    elif algorithm in algorithm_library_map["surprise_algorithms"]:
        return surprise_tb(data, algorithm)
    elif algorithm in algorithm_library_map["myfm_algorithms"]:
        return myfm_tb(data, algorithm)
    elif algorithm in algorithm_library_map["sklearn_algorithms"]:
        return sklearn_tb(data, algorithm)
    elif algorithm in algorithm_library_map["xgboost_algorithms"]:
        return xgboost_tb(data, algorithm)
    else:
        print(f"Exception: The algorithm {Algorithm.name} is unsupported.")
        raise Exception
