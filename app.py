from flask import Flask, jsonify, render_template, request

from lib.constants import DOMAIN, OBJECTIVE_FUNCTIONS
from lib.evolutionary_algorithms import cma, es
from lib.fcm import call_fcm, call_fcm_cma

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("base.html")


@app.route("/es", methods=["POST", "GET"])
def es_page():
    if request.method == "GET":
        return render_template("es.html", title="Evolution Strategy Algorithms")
    elif request.method == "POST":
        data = request.get_json()
        dimension = int(data["dimension"])  # [2, 32]
        generation_size = int(data["generation_size"])  # [100, 10000]
        offspring_size = int(data["offspring_size"])  # [5, 5000]
        parent_size = int(data["parent_size"])  # [5, 5000]
        step_size = float(data["step_size"])  # [0.1, 1.00]
        survivor_selection = data["survivor_selection"]
        benchmark_function = data["benchmark_function"]
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


@app.route("/cma", methods=["POST", "GET"])
def cma_page():
    if request.method == "GET":
        return render_template("cma.html", title="CMA-ES Algorithm")
    elif request.method == "POST":
        data = request.get_json()
        dimension = int(data["dimension"])  # [2, 32]
        benchmark_function = data["benchmark_function"]
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


@app.route("/fcm", methods=["POST", "GET"])
def fcm_page():
    if request.method == "GET":
        return render_template("fcm.html")
    elif request.method == "POST":
        data = request.get_json()
        n_iter = int(data["n_iter"])  # [100, 10000]
        m = float(data["m"])  # [1.5, 10]
        dataset_name = data["dataset_name"]
        table_data, chart_data = call_fcm(n_iter, m, dataset_name)
        result = {
            "table_data": table_data,
            "chart_data": chart_data,
        }
        return jsonify(result)


@app.route("/fcm-cma", methods=["POST", "GET"])
def fcm_cma_page():
    if request.method == "GET":
        return render_template("fcm-cma.html")
    elif request.method == "POST":
        data = request.get_json()
        m = float(data["m"])  # [1.5, 10]
        l = data["l"]  # zero, linear, success_rule
        dataset_name = data["dataset_name"]
        table_data, chart_data = call_fcm_cma(m, l, dataset_name)
        result = {
            "table_data": table_data,
            "chart_data": chart_data,
        }
        return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
