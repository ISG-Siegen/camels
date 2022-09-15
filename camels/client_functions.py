from pathlib import Path
import argparse
import os

from camels.client.client_routine import c_populate_database, c_evaluate_algorithms, c_train_meta_learner, \
    c_predict_with_meta_learner
from camels.client.client_database_identifier import Metric, Algorithm, Learner, Task
from camels.data_loader.data_processing import load_data, all_selectors

file_path = Path(os.path.dirname(os.path.abspath(__file__)))

# enables command line arguments to run client routine
parser = argparse.ArgumentParser("CaMeLS client routine")
parser.add_argument('--populate', dest='populate', action='store_true')
parser.add_argument('--evaluate_single', nargs='*',
                    help='Pass the following: data_set_category data_set algorithm_name')
parser.add_argument('--evaluate_all', dest='evaluate_all', action='store_true')
parser.add_argument('--meta_learner', dest='meta_learner', action='store_true')
parser.add_argument('--predict_single', nargs='*',
                    help='Pass the following: data_set_category data_set metric_name learner_name')
args = parser.parse_args()

if args.populate:
    # populate the database with tables and administrative entries
    c_populate_database()

if args.evaluate_single is not None:
    # evaluate single data on single algorithm based on command line input
    for data_df, stats in load_data([(args.evaluate_single[0], args.evaluate_single[1])],
                                    Path(file_path / "data_loader/data_sets/"), False, False):
        c_evaluate_algorithms(
            algos=[Algorithm[args.evaluate_single[2]]],
            metrics=[
                Metric.FitTime,
                Metric.PredictTime,
                Metric.RootMeanSquaredError,
                Metric.MeanAbsoluteError
            ],
            task=Task.ExplicitRatingPrediction,
            data=data_df
        )
elif args.evaluate_all:
    # evaluate algorithms locally and donate metadata sets to the server
    for data_df, stats in load_data(all_selectors, Path(file_path / "data_loader/data_sets/"), False, False):
        c_evaluate_algorithms(
            algos=[
                Algorithm.Bias,
                Algorithm.ItemItem,
                Algorithm.UserUser,
                Algorithm.BiasedMF,
                Algorithm.ImplicitMF,
                Algorithm.BiasedSVD,
                Algorithm.FunkSVD,
                Algorithm.NormalPredictor,
                Algorithm.Baseline,
                Algorithm.KNNBasic,
                Algorithm.KNNWithMeans,
                Algorithm.KNNWithZScore,
                Algorithm.KNNBaseline,
                Algorithm.SVD,
                Algorithm.SVDpp,
                Algorithm.NMF,
                Algorithm.SlopeOne,
                Algorithm.CoClustering,
                Algorithm.BayesianFactorizationMachine,
                Algorithm.LinearRegression,
                Algorithm.RandomForestRegression,
                Algorithm.HistogramGradientBoostingRegression,
                Algorithm.XGBoostRegression
            ],
            metrics=[
                Metric.FitTime,
                Metric.PredictTime,
                Metric.RootMeanSquaredError,
                Metric.MeanAbsoluteError
            ],
            task=Task.ExplicitRatingPrediction,
            data=data_df
        )

if args.meta_learner:
    # train a meta-learner on the server
    c_train_meta_learner(metrics=[Metric.RootMeanSquaredError, Metric.MeanAbsoluteError],
                         tasks=[Task.ExplicitRatingPrediction], learners=[Learner.RandomForest])

if args.predict_single is not None:
    # predict metadata performance with meta-learner and return best model
    for data_df, stats in load_data([(args.predict_single[0], args.predict_single[1])],
                                    Path(file_path / "data_loader/data_sets/"), False, False):
        c_predict_with_meta_learner(metric=Metric[args.predict_single[2]],
                                    task=Task.ExplicitRatingPrediction,
                                    learner=Learner[args.predict_single[3]],
                                    data=data_df)
