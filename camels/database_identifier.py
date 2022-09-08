from enum import Enum


class Algorithm(Enum):
    UserUser = 0
    ItemItem = 1
    BiasedMF = 2
    BiasedSVD = 3
    FunkSVD = 4
    Bias = 5
    NormalPredictor = 6
    Baseline = 7
    KNNBasic = 8
    KNNWithMeans = 9
    KNNWithZScore = 10
    KNNBaseline = 11
    SVD = 12
    NMF = 13
    SlopeOne = 14
    CoClustering = 15


class Task(Enum):
    ERP = 0
    IRP = 1


class Metric(Enum):
    Runtime = 0
    NMAE = 1
    NRMSE = 2
    MAE = 3
    RMSE = 4


lenskit_algorithms = [Algorithm.UserUser, Algorithm.ItemItem, Algorithm.BiasedMF, Algorithm.BiasedSVD,
                      Algorithm.FunkSVD, Algorithm.Bias]

surprise_algorithms = [Algorithm.NormalPredictor, Algorithm.Baseline, Algorithm.KNNBasic, Algorithm.KNNWithMeans,
                       Algorithm.KNNWithZScore, Algorithm.KNNBaseline, Algorithm.SVD, Algorithm.NMF,
                       Algorithm.SlopeOne, Algorithm.CoClustering]
