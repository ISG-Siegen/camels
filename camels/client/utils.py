from camels.database_identifier import Algorithm, lenskit_algorithms, surprise_algorithms
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.als import BiasedMF
from lenskit.algorithms.funksvd import FunkSVD
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.svd import BiasedSVD
from lenskit.algorithms.basic import Fallback
from surprise import SVD
from surprise import KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore
from surprise import CoClustering, BaselineOnly, SlopeOne, NMF, NormalPredictor


def get_base_model(algorithm):
    recommender = None
    if algorithm in lenskit_algorithms:
        if algorithm == Algorithm.UserUser:
            recommender = Fallback(UserUser(100), Bias())
        elif algorithm == Algorithm.ItemItem:
            recommender = Fallback(ItemItem(100), Bias())
        elif algorithm == Algorithm.BiasedMF:
            recommender = Fallback(BiasedMF(100), Bias())
        elif algorithm == Algorithm.FunkSVD:
            recommender = Fallback(FunkSVD(100), Bias())
        elif algorithm == Algorithm.BiasedSVD:
            recommender = Fallback(BiasedSVD(100), Bias())
        elif algorithm == Algorithm.Bias:
            recommender = Bias()
        else:
            recommender = None
    elif algorithm in surprise_algorithms:
        if algorithm == Algorithm.NormalPredictor:
            recommender = NormalPredictor()
        elif algorithm == Algorithm.Baseline:
            recommender = BaselineOnly()
        elif algorithm == Algorithm.KNNBasic:
            recommender = KNNBasic()
        elif algorithm == Algorithm.KNNWithMeans:
            recommender = KNNWithMeans()
        elif algorithm == Algorithm.KNNWithZScore:
            recommender = KNNWithZScore()
        elif algorithm == Algorithm.KNNBaseline:
            recommender = KNNBaseline()
        elif algorithm == Algorithm.SVD:
            recommender = SVD()
        elif algorithm == Algorithm.NMF:
            recommender = NMF()
        elif algorithm == Algorithm.SlopeOne:
            recommender = SlopeOne()
        elif algorithm == Algorithm.CoClustering:
            recommender = CoClustering()
        else:
            recommender = None

    return recommender
