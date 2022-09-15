from typing import Union, Tuple
import pandas as pd
import time

from camels.client.client_database_identifier import Algorithm

from surprise import Dataset, Reader
from surprise import SVD, SVDpp, KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore, CoClustering, BaselineOnly, \
    SlopeOne, NMF, NormalPredictor


def get_base_model(algorithm: Algorithm) -> Union[
    NormalPredictor,
    BaselineOnly,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    KNNBaseline,
    SVD,
    SVDpp,
    NMF,
    SlopeOne,
    CoClustering
]:
    if algorithm == Algorithm.NormalPredictor:
        return NormalPredictor()
    elif algorithm == Algorithm.Baseline:
        return BaselineOnly()
    elif algorithm == Algorithm.KNNBasic:
        return KNNBasic()
    elif algorithm == Algorithm.KNNWithMeans:
        return KNNWithMeans()
    elif algorithm == Algorithm.KNNWithZScore:
        return KNNWithZScore()
    elif algorithm == Algorithm.KNNBaseline:
        return KNNBaseline()
    elif algorithm == Algorithm.SVD:
        return SVD()
    elif algorithm == Algorithm.SVDpp:
        return SVDpp()
    elif algorithm == Algorithm.NMF:
        return NMF()
    elif algorithm == Algorithm.SlopeOne:
        return SlopeOne()
    elif algorithm == Algorithm.CoClustering:
        return CoClustering()
    else:
        print("Exception: The algorithm could not be identified.")
        raise Exception


def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame, algorithm: Algorithm) -> Tuple[pd.DataFrame, float, float]:
    # choose model
    print(f"Evaluating lenskit algorithm {algorithm.name}.")
    recommender = get_base_model(algorithm)

    reader = Reader(rating_scale=(train["rating"].min(), train["rating"].max()))
    train = Dataset.load_from_df(train, reader).build_full_trainset()
    fit_start_time = time.time()
    recommender.fit(train)
    fit_duration = time.time() - fit_start_time
    print(f"Algorithm fitted in {fit_duration} seconds.")
    predict_start_time = time.time()
    prediction = [recommender.predict(getattr(row, "user"), getattr(row, "item"))
                  for row in test.itertuples()]
    prediction = pd.DataFrame(prediction)["est"]
    predict_duration = time.time() - predict_start_time
    print(f"Algorithm predicted in {predict_duration} seconds.")

    return prediction, fit_duration, predict_duration


def train_best_model(data: pd.DataFrame, algorithm: Algorithm) -> Union[
    NormalPredictor,
    BaselineOnly,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    KNNBaseline,
    SVD,
    SVDpp,
    NMF,
    SlopeOne,
    CoClustering
]:
    recommender = get_base_model(algorithm)
    reader = Reader(rating_scale=(data["rating"].min(), data["rating"].max()))
    data = Dataset.load_from_df(data, reader).build_full_trainset()
    if recommender is not None:
        print(f"Training best algorithm {algorithm.name} on supplied data.")
        recommender.fit(data)
        print(f"Done.")
        return recommender
    else:
        print(f"Exception: Algorithm {algorithm.name} could not be built.")
        raise Exception
