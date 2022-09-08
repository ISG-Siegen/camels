from flask import Flask, jsonify, request
from database_manager import populate_database
from metadata_manager import check_hash, write_metadata
from run_manager import check_data_status, save_runs
from metamodel_manager import train_meta_learner, predict_with_meta_learner
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)


@app.route("/populate_database", methods=["POST"])
def s_populate_database():
    ret = populate_database()
    return jsonify(ret)


@app.route("/check_hash", methods=["POST"])
def s_get_metadata():
    ret = check_hash(request.values['hash'])
    return jsonify(ret)


@app.route("/save_metadata", methods=["POST"])
def s_save_metadata():
    ret = write_metadata(request.values['metadata'])
    return jsonify(ret)


@app.route("/check_data_status", methods=["POST"])
def s_check_data_status():
    ret = check_data_status(request.values['metadata_id'], request.values['algo_ids'], request.values['task_id'],
                            request.values['metric_ids'])
    return jsonify(ret)


@app.route("/save_runs", methods=["POST"])
def s_save_runs():
    ret = save_runs(request.values['evals'])
    return jsonify(ret)


@app.route("/train_meta_learner", methods=["POST"])
def s_train_meta_learner():
    ret = train_meta_learner(request.values['metric_ids'], request.values['task_ids'], request.values['learners'])
    return jsonify(ret)


@app.route("/predict_with_meta_learner", methods=["POST"])
def s_predict_with_meta_learner():
    ret = predict_with_meta_learner(request.values['meta_data'], request.values['metric_id'], request.values['learner'],
                                    request.values['task_id'])
    return jsonify(ret)
