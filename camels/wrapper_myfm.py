from typing import Union, Tuple
import pandas as pd
import time

from camels.client.client_database_identifier import Algorithm

from myfm import MyFMRegressor
from sklearn.preprocessing import OneHotEncoder


def get_base_model(algorithm: Algorithm) -> Union[MyFMRegressor]:
    if algorithm == Algorithm.BayesianFactorizationMachine:
        return MyFMRegressor(rank=10)
    else:
        print("Exception: The algorithm could not be identified.")
        raise Exception


def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame, algorithm: Algorithm) -> Tuple[pd.DataFrame, float, float]:
    # choose model
    print(f"Evaluating myfm algorithm {algorithm.name}.")
    recommender = get_base_model(algorithm)

    # data has to be one-hot encoded
    ohe = OneHotEncoder(handle_unknown='ignore')
    # start fitting
    fit_start_time = time.time()
    recommender.fit(ohe.fit_transform(train.drop(columns=["rating"])), train["rating"].values, n_iter=200,
                    n_kept_samples=200)
    fit_duration = time.time() - fit_start_time
    print(f"Algorithm fitted in {fit_duration} seconds.")

    # predict
    predict_start_time = time.time()
    prediction = recommender.predict(ohe.transform(test.drop(columns=["rating"])))
    predict_duration = time.time() - predict_start_time
    print(f"Algorithm predicted in {predict_duration} seconds.")

    return prediction, fit_duration, predict_duration


def train_best_model(data: pd.DataFrame, algorithm: Algorithm) -> MyFMRegressor:
    # train the best recommender based on predictions
    recommender = get_base_model(algorithm)
    if recommender is not None:
        print(f"Training best algorithm {algorithm.name} on supplied data.")
        ohe = OneHotEncoder(handle_unknown='ignore')
        recommender.fit(ohe.fit_transform(data.drop(columns=["rating"])), data["rating"].values, n_iter=200,
                        n_kept_samples=200)
        print(f"Done.")
        return recommender
    else:
        print(f"Exception: Algorithm {algorithm.name} could not be built.")
        raise Exception
