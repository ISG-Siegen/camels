from enum import Enum, auto


class Algorithm(Enum):
    Bias = auto()
    ItemItem = auto()
    UserUser = auto()
    BiasedMF = auto()
    ImplicitMF = auto()
    BiasedSVD = auto()
    FunkSVD = auto()
    NormalPredictor = auto()
    Baseline = auto()
    KNNBasic = auto()
    KNNWithMeans = auto()
    KNNWithZScore = auto()
    KNNBaseline = auto()
    SVD = auto()
    SVDpp = auto()
    NMF = auto()
    SlopeOne = auto()
    CoClustering = auto()
    BayesianFactorizationMachine = auto()
    LinearRegression = auto()
    RandomForestRegression = auto()
    HistogramGradientBoostingRegression = auto()
    XGBoostRegression = auto()


class Task(Enum):
    ExplicitRatingPrediction = auto()
    ImplicitRankingPrediction = auto()


class Metric(Enum):
    FitTime = auto()
    PredictTime = auto()
    RootMeanSquaredError = auto()
    MeanAbsoluteError = auto()


class Learner(Enum):
    RandomForest = auto()
    LinearRegression = auto()


class Metadata(Enum):
    NumUsers = auto()
    NumItems = auto()
    NumInstances = auto()
    Sparsity = auto()
    Density = auto()
    MinRating = auto()
    MaxRating = auto()
    MeanRating = auto()
    MedianRating = auto()
    ModeRating = auto()
    NormalizedMeanRating = auto()
    NormalizedMedianRating = auto()
    NormalizedModeRating = auto()
    UserItemRatio = auto()
    ItemUserRatio = auto()
    HighestNumRatingBySingleUser = auto()
    LowestNumRatingBySingleUser = auto()
    HighestNumRatingOnSingleItem = auto()
    LowestNumRatingOnSingleItem = auto()
    MeanNumRatingsByUser = auto()
    MeanNumRatingsOnItem = auto()
    RatingSkew = auto()
    RatingKurtosis = auto()
    RatingsStandardVariation = auto()
    RatingVariance = auto()
    RatingEntropy = auto()
    NumPossibleRatings = auto()
    RatingAverageIncrement = auto()


algorithm_library_map = {
    "lenskit_algorithms": [Algorithm.Bias,
                           Algorithm.ItemItem,
                           Algorithm.UserUser,
                           Algorithm.BiasedMF,
                           Algorithm.ImplicitMF,
                           Algorithm.BiasedSVD,
                           Algorithm.FunkSVD],
    "surprise_algorithms": [Algorithm.NormalPredictor,
                            Algorithm.Baseline,
                            Algorithm.KNNBasic,
                            Algorithm.KNNWithMeans,
                            Algorithm.KNNWithZScore,
                            Algorithm.KNNBaseline,
                            Algorithm.SVD,
                            Algorithm.SVDpp,
                            Algorithm.NMF,
                            Algorithm.SlopeOne,
                            Algorithm.CoClustering],
    "myfm_algorithms": [Algorithm.BayesianFactorizationMachine],
    "sklearn_algorithms": [Algorithm.LinearRegression,
                           Algorithm.RandomForestRegression,
                           Algorithm.HistogramGradientBoostingRegression],
    "xgboost_algorithms": [Algorithm.XGBoostRegression]
}
