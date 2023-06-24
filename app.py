import numpy as np
from flask import Flask, jsonify, render_template, request

from code.constants import DOMAIN, OBJECTIVE_FUNCTIONS
from code.evolutionary_algorithms import cma, es
from code.fcm import call_fcm, call_fcm_cma

app = Flask(__name__)


@app.route("/")
def home():
    # Your Python script logic goes here
    message = "Hello from Python!"
    return render_template("home.html", message=message)


@app.route("/es")
def get_es():
    return render_template("es.html")


@app.route("/es", methods=["POST"])
def submit_es():
    data = request.get_json()

    dimension = int(data["dimension"])  # [2, 32]
    generation_size = int(data["generation_size"])  # [100, 10000]
    offspring_size = int(data["offspring_size"])  # [5, 5000]
    parent_size = int(data["parent_size"])  # [5, 5000]
    step_size = float(data["step_size"])  # [0.01, 1.00]
    survivor_selection = data["survivor_selection"]  # comma, plus
    benchmark_function = data["benchmark_function"]  # sphere, rastrigin, ackley

    table_data, chart_data = es(
        domain=DOMAIN[benchmark_function],
        objective_function=OBJECTIVE_FUNCTIONS[benchmark_function],
        keep_parent=True if survivor_selection == "plus" else False,
        n_generation=generation_size,
        SIGMA=step_size,
        LAMBDA=offspring_size,
        MU=parent_size,
        DIM=dimension,
    )

    result = {
        "table_data": table_data,
        "chart_data": chart_data,
    }

    return jsonify(result)


@app.route("/cma")
def get_cma():
    return render_template("cma.html")


@app.route("/cma", methods=["POST"])
def submit_cma():
    data = request.get_json()

    dimension = int(data["dimension"])  # [2, 32]
    benchmark_function = data["benchmark_function"]  # sphere, rastrigin, ackley

    table_data, chart_data = cma(
        domain=DOMAIN[benchmark_function],
        objective_function=OBJECTIVE_FUNCTIONS[benchmark_function],
        dim=dimension,
    )

    result = {
        "table_data": table_data,
        "chart_data": chart_data,
    }

    return jsonify(result)


@app.route("/fcm")
def get_fcm():
    return render_template("fcm.html")


@app.route("/fcm", methods=["POST"])
def submit_fcm():
    data = request.get_json()

    n_iter = int(data["n_iter"])  # [100, 10000]
    m = float(data["m"])  # [2, 101]
    dataset_name = data[
        "dataset_name"
    ]  # Hepta, Tetra, Chainlink, Atom, GolfBall, Lsun, EngyTime, Target, TwoDiamonds, WingNut, MySet1, MySet2

    table_data, chart_data = call_fcm(n_iter, m, dataset_name)

    result = {
        "table_data": table_data,
        "chart_data": chart_data,
    }

    return jsonify(result)


@app.route("/fcm-cma")
def get_fcm_cma():
    return render_template("fcm-cma.html")


@app.route("/fcm-cma", methods=["POST"])
def submit_fcm_cma():
    data = request.get_json()

    n_iter = int(data["n_iter"])  # [100, 10000]
    m = float(data["m"])  # [2, 101]
    l = data["l"]  # zero, linear, success_rule
    dataset_name = data[
        "dataset_name"
    ]  # Hepta, Tetra, Chainlink, Atom, GolfBall, Lsun, EngyTime, Target, TwoDiamonds, WingNut, MySet1, MySet2

    table_data, chart_data = call_fcm_cma(n_iter, m, l, dataset_name)

    result = {
        "table_data": table_data,
        "chart_data": chart_data,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run()
