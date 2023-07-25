import numpy as np


DOMAIN = {"sphere": [-5.12, 5.12], "rastrigin": [-5.12, 5.12], "ackley": [-32.768, 32.768]}


def SPHERE(x):
    return np.sum(np.square(x), axis=2)


def RASTRIGIN(x):
    d = x.shape[2]
    return 10 * d + np.sum(np.square(x) - 10 * (np.cos(2 * np.pi * x)), axis=2)


def ACKLEY(x):
    d = x.shape[2]
    return (
        20
        - 20 * np.exp(-0.2 * np.sqrt((1 / d) * np.sum(np.square(x), axis=2)))
        + np.e
        - np.exp((1 / d) * np.sum((np.cos(2 * np.pi * x)), axis=2))
    )


OBJECTIVE_FUNCTIONS = {"sphere": SPHERE, "rastrigin": RASTRIGIN, "ackley": ACKLEY}

DATASET_N_CLUSTERS = {
    "Hepta": 7,
    "Tetra": 4,
    "Chainlink": 2,
    "Atom": 2,
    "GolfBall": 1,
    "Lsun": 3,
    "EngyTime": 2,
    "Target": 6,
    "TwoDiamonds": 2,
    "WingNut": 2,
    "MySet1": 4,
    "MySet2": 4,
}

DATASET_DIMENSION = {
    "Hepta": 3,
    "Tetra": 3,
    "Chainlink": 3,
    "Atom": 3,
    "GolfBall": 3,
    "Lsun": 2,
    "EngyTime": 2,
    "Target": 2,
    "TwoDiamonds": 2,
    "WingNut": 2,
    "MySet1": 2,
    "MySet2": 2,
}

DATASET_PATH = "D:/Uni/Term 9/Final Project/Webpage/static/dataset/FCPS/Sets/"

DATASET_LABELS_PATH = "D:/Uni/Term 9/Final Project/Webpage/static/dataset/FCPS/Labels/"