import base64
import io

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg


def es_plot(x, y, title):
    x1 = x[:, :, 0]
    x2 = x[:, :, 1]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    plt.plot(x1, y, "b-o", label="x1")
    plt.plot(x2, y, "r-o", label="x2")
    ax.legend()
    png = __draw_canvas(fig)
    plt.close("all")
    return png


def contour_plot(domain, level, objective_function, given_X, title):
    d_min = np.min(domain)
    d_max = np.max(domain)
    domain = np.arange(d_min, d_max, 0.1)
    x_1, x_2 = np.meshgrid(domain, domain)
    X = np.dstack((x_1, x_2))
    y = objective_function(X)
    fig = plt.figure(figsize=(6, 6))
    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1.set_title("Contour of " + title)
    ax_1.contour(x_1, x_2, y, cmap="turbo", levels=level)
    in_domain = (given_X > d_min + 0.1) * (given_X < d_max - 0.1)
    given_X = np.delete(given_X, np.where(in_domain == False)[0], axis=0)
    plt.plot(given_X[:, :, 0], given_X[:, :, 1], "r+")
    png = __draw_canvas(fig)
    plt.close("all")
    return png


def sigma_plot(sigma, generation, d, objective_function):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Step-Size of " + objective_function.__name__ + " ,d=" + str(d))
    plt.xlabel("generation")
    plt.ylabel("sigma")
    plt.plot(range(generation), sigma, "b--.")
    png = __draw_canvas(fig)
    plt.close("all")
    return png


def visualize_fcm_result(dataset, dataset_clusters, label, dimension, is_real_label):
    fig = plt.figure(figsize=(6, 6))
    projection = "3d" if dimension == 3 else None
    title = f"Real labels, clusters={dataset_clusters}" if is_real_label else f"Predicted labels"
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_title(title)
    if dimension == 3:
        ax.scatter(dataset["C1"], dataset["C2"], dataset["C3"], c=label)
    else:
        ax.scatter(dataset["C1"], dataset["C2"], c=label)
    plt.tight_layout()
    png = __draw_canvas(fig)
    plt.close("all")
    return png


def visualize_cost_function(J, j):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    if j == 0:
        ax.plot(range(len(J)), J, "r-")
        ax.set_xlabel("generation")
        ax.set_title("L parameter")
    else:
        ax.plot(range(len(J)), J, "b-")
        ax.set_xlabel("generation")
        ax.set_title("cost function")
    png = __draw_canvas(fig)
    plt.close("all")
    return png


def __draw_canvas(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    # Get the PNG data as a byte buffer
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    png_data = buf.getvalue()
    return base64.b64encode(png_data).decode("utf-8")
