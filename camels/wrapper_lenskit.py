from typing import Union, Tuple
import pandas as pd
import time

from camels.client.client_database_identifier import Algorithm

from lenskit.algorithms import Recommender
from lenskit.batch import predict
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.als import BiasedMF
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.svd import BiasedSVD
from lenskit.algorithms.funksvd import FunkSVD
from lenskit.algorithms.basic import Fallback


def get_base_model(algorithm: Algorithm) -> Union[Bias, Fallback]:
    if algorithm == Algorithm.Bias:
        return Bias()
    elif algorithm == Algorithm.ItemItem:
        return Fallback(ItemItem(20), Bias())
    elif algorithm == Algorithm.UserUser:
        return Fallback(UserUser(20), Bias())
    elif algorithm == Algorithm.BiasedMF:
        return Fallback(BiasedMF(20), Bias())
    elif algorithm == Algorithm.ImplicitMF:
        return Fallback(ImplicitMF(20), Bias())
    elif algorithm == Algorithm.BiasedSVD:
        return Fallback(BiasedSVD(20), Bias())
    elif algorithm == Algorithm.FunkSVD:
        return Fallback(FunkSVD(20), Bias())
    else:
        print("Exception: The algorithm could not be identified.")
        raise Exception


def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame, algorithm: Algorithm) -> Tuple[pd.DataFrame, float, float]:
    # choose model
    print(f"Evaluating lenskit algorithm {algorithm.name}.")
    recommender = Recommender.adapt(get_base_model(algorithm))

    fit_start_time = time.time()
    recommender.fit(train)
    fit_duration = time.time() - fit_start_time
    print(f"Algorithm fitted in {fit_duration} seconds.")
    predict_start_time = time.time()
    # batch predict
    # prediction = predict(recommender, d_test.drop(columns="rating"), n_jobs=2)["prediction"]
    # sequential predict
    prediction = recommender.predict(test.drop(columns="rating"))
    predict_duration = time.time() - predict_start_time
    print(f"Algorithm predicted in {predict_duration} seconds.")

    return prediction, fit_duration, predict_duration


def train_best_model(data: pd.DataFrame, algorithm: Algorithm) -> Recommender:
    # train the best recommender based on predictions
    recommender = Recommender.adapt(get_base_model(algorithm))
    if recommender is not None:
        recommender = Recommender.adapt(recommender)
        print(f"Training best algorithm {algorithm.name} on supplied data.")
        recommender.fit(data)
        print(f"Done.")
        return recommender
    else:
        print(f"Exception: Algorithm {algorithm.name} could not be built.")
        raise Exception
