import time

import numpy as np
import pandas as pd

from sklearn.metrics import rand_score

from typing import Optional
from pydantic import BaseModel, Extra, Field, validate_arguments

from plots import visualize_fcm_result
from code.constants import DATASET_N_CLUSTERS, DATASET_DIMENSION, DATASET_PATH, DATASET_LABELS_PATH


class FCM(BaseModel):
    # Step 1
    n_clusters: int = Field(5, ge=1, le=100)
    m: float = Field(2.0, ge=1.0)
    max_iter: int = Field(150, ge=1, le=1000)
    error: float = Field(1e-5, ge=1e-9)
    random_state: Optional[int] = None
    trained: bool = Field(False, const=True)

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True

    @validate_arguments
    def fit(self, X):
        n_samples = X.shape[0]

        # Construct a random number generator
        self.rng = np.random.default_rng(self.random_state)

        # Step 2
        # Initialize u randomly
        # Draw samples from a uniform distribution
        # Samples are uniformly distributed over the interval [0.0, 1.0)
        # (Equation1) is not satisfied here yet, raw sum of u is not equal to 1
        # Example for a sample with two clusters:
        # [.2, .3] -> .2 + .3 = .5 != 1
        self.u = self.rng.uniform(size=(n_samples, self.n_clusters))

        # Row sum of u must be equal to 1, based on (Equation1)
        # Each row shows membership (u) of different clusters for a sample
        # Example for a sample with two clusters:
        # [.2, .3] -> [.2/(.2+.3), .3/(.2+.3)] -> [.2/.5, .3/.5] -> [.4, .6] -> .4 + .6 = 1
        self.u = self.u / np.tile(self.u.sum(axis=1)[np.newaxis].T, self.n_clusters)

        # While ||u(i+1) - u(i)|| >= error, update centers and u
        # || || is Euclidean Norm here
        for _ in range(self.max_iter):
            u_old = self.u.copy()

            # Step 3
            # Calculate centers using (Equation 3)
            self._centers = FCM._next_centers(X, self.u, self.m)

            # Step 4
            # Update U using (Equation 4)
            self.u = self.soft_predict(X)

            # Step 5
            # Stopping rule
            if np.linalg.norm(self.u - u_old) < self.error:
                break

        self.trained = True

    def soft_predict(self, X):
        # Calculate the distance of 3 samples and 2 clusters, raise all distances to power 2/(m-1)
        # [[d11=d11^(2/(m-1)), d12=d12^(2/(m-1))],
        #  [d21=d21^(2/(m-1)), d22=d22^(2/(m-1))],
        #  [d31=d31^(2/(m-1)), d32=d32^(2/(m-1))]]
        temp = FCM._dist(X, self._centers) ** float(2 / (self.m - 1))

        # Expansion of (Equation 4) for i=2, j=2 (2 clusters), and k=3 (3 samples) is:
        # [[1/(d11/d11+d12/d11), 1/(d11/d12+d12/d12)],
        #  [1/(d21/d21+d22/d21), 1/(d21/d22+d22/d22)],
        #  [1/(d31/d31+d32/d31), 1/(d31/d32+d32/d32)]]

        # [[[d11, d12],
        #   [d11, d12]],
        #  [[d21, d22],
        #   [d21, d22]],
        #  [[d31, d32],
        #   [d31, d32]]]
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)

        # [[[d11],
        #   [d12]],
        #  [[d21],
        #   [d22]],
        #  [[d31],
        #   [d32]]]
        temp = temp[:, :, np.newaxis]

        # [[[d11/d11, d12/d11],
        #   [d11/d12, d12/d12]],
        #  [[d21/d21, d22/d21],
        #   [d21/d22, d22/d22]],
        #  [[d31/d31, d32/d31],
        #   [d31/d32, d32/d32]]]
        denominator_ = temp / denominator_

        # [[d11/d11+d12/d11, d11/d12+d12/d12],
        #  [d21/d21+d22/d21, d21/d22+d22/d22],
        #  [d31/d31+d32/d31, d31/d33+d32/d32]]
        denominator_ = denominator_.sum(2)

        # [[1/(d11/d11+d12/d11), 1/(d11/d12+d12/d12)],
        #  [1/(d21/d21+d22/d21), 1/(d21/d22+d22/d22)],
        #  [1/(d31/d31+d32/d31), 1/(d31/d32+d32/d32)]]
        return 1 / denominator_

    @validate_arguments
    def predict(self, X):
        if self.is_trained():
            # If X is 1-D, increase its dimension to 2-D
            X = np.expand_dims(X, axis=0) if len(X.shape) == 1 else X

            # Return the index of the cluster with higher membership for each instance
            # In other words, index of the cluster each sample belongs to
            return self.soft_predict(X).argmax(axis=-1)

    def is_trained(self):
        if self.trained:
            return True
        raise ReferenceError("You need to train the model. Run `.fit()` method to this.")

    @staticmethod
    def _dist(A, B):
        # Compute the Euclidean Distance of two matrices
        # Example for 3 samples with 2 clusters:
        # [[[x11, x12]],
        #  [[x21, x22]],
        #  [[x31, x32]]]
        A = A[:, None, :]

        # [[[d11_1=(x11-c11)^2, d11_2=(x12-c12)^2],
        #   [d12_1=(x11-c21)^2, d12_2=(x12-c22)^2]],
        #  [[d21_1=(x21-c11)^2, d21_2=(x22-c12)^2],
        #   [d22_1=(x21-c21)^2, d22_2=(x22-c22)^2]],
        #  [[d31_1=(x31-c11)^2, d31_2=(x32-c12)^2],
        #   [d32_1=(x31-c21)^2, d32_2=(x32-c22)^2]]]
        dist = (A - B) ** 2

        # [[d11=√(d11_1+d11_2), d12=√(d12_1+d12_2)],
        #  [d21=√(d21_1+d21_2), d22=√(d22_1+d22_2)],
        #  [d31=√(d31_1+d31_2), d32=√(d32_1+d32_2)]]
        return np.sqrt(np.einsum("ijk->ij", dist))

    @staticmethod
    def _next_centers(X, u, m):
        # [[u11=u11^m, u12=u12^m],
        #  [u21=u21^m, u22=u22^m],
        #  [u31=u31^m, u32=u32^m]]
        um = u**m

        # Expansion of (Equation 3) for i=2 (2 clusters), and k=3 (3 samples) is:
        # [[(x11.u11+x21.u21+x31.u31)/(u11+u21+u31), (x12.u11+x22.u21+x32.u31)/(u11+u21+u31)],
        #  [(x11.u12+x21.u22+x31.u32)/(u12+u22+u32), (x12.u12+x22.u22+x32.u32)/(u12+u22+u32)]]

        # Multiple matrices X.T and u
        # [[x11.u11+x21.u21+x31.u31, x11.u12+x21.u22+x31.u32],
        #  [x12.u11+x22.u21+x32.u31, x12.u12+x22.u22+x32.u32]]
        uX = X.T @ um

        # ([[(x11.u11+x21.u21+x31.u31)/(u11+u21+u31), (x11.u12+x21.u22+x31.u32)/(u12+u22+u32)],
        #   [(x12.u11+x22.u21+x32.u31)/(u11+u21+u31), (x12.u12+x22.u22+x32.u32)/(u12+u22+u32)]]).T
        return (uX / np.sum(um, axis=0)).T

    @property
    def centers(self):
        if self.is_trained():
            return self._centers

    @property
    def partition_coefficient(self):
        # Cluster validity functional
        # The PC (Partition Coefficient) is defined on the range from 0 to 1, with 1 being best
        # It is a metric which tells us how cleanly our data is described by a certain model
        if self.is_trained():
            # (Equation 5)
            return np.mean(self.u**2)

    @property
    def partition_entropy_coefficient(self):
        # Cluster validity functional
        # The PE (Partition Entropy) is defined on the range from 0 to log(c), with 0 being best
        # It is a metric which tells us how cleanly our data is described by a certain model
        if self.is_trained():
            # (Equation 6)
            return -np.mean(self.u * np.log2(self.u))


class FCM_CMA(BaseModel):
    # Step 1
    n_clusters: int = Field(5, ge=1, le=100)
    m: float = Field(2.0, ge=1.0)
    max_iter: int = Field(150, ge=1, le=1000)
    error: float = Field(1e-5, ge=1e-9)
    random_state: Optional[int] = None
    trained: bool = Field(False, const=True)


def call_fcm(n_iter, m, dataset_name):
    dataset = get_dataset(dataset_name)
    dataset_labels = get_dataset_labels(dataset_name)
    dataset_clusters = get_dataset_clusters(dataset_name)
    dataset_dimension = get_dataset_dimension(dataset_name)

    start_time = time.time()
    fcm = FCM(n_clusters=dataset_clusters, m=m, max_iter=n_iter, error=1e-8)
    fcm.fit(dataset.to_numpy())
    fcm_labels = fcm.predict(dataset.to_numpy())
    exec_time = format(time.time() - start_time, ".4f")

    fc = format(fcm.partition_coefficient, ".2f")
    hc = format(fcm.partition_entropy_coefficient, ".2f")

    rand_index = format(rand_score(dataset_labels, fcm_labels), ".2f")

    dataset_png = visualize_fcm_result(
        dataset, dataset_clusters, dataset_labels, dataset_dimension, True
    )

    algorithm_png = visualize_fcm_result(
        dataset, dataset_clusters, fcm_labels, dataset_dimension, False
    )

    table_data = {
        "rand_index": str(rand_index),
        "fc": str(fc),
        "hc": str(hc),
        "exec_time": str(exec_time),
    }
    chart_data = {"dataset_png": dataset_png, "algorithm_png": algorithm_png}

    return table_data, chart_data


def call_fcm_cma(n_iter, m, l, dataset_name):
    rand_index = 0
    fc = 0
    hc = 0
    cost_value = 0
    exec_time = 0

    dataset_png = ""
    algorithm_png = ""
    cost_function = ""
    l_param = ""

    table_data = {
        "rand_index": str(rand_index),
        "fc": str(fc),
        "hc": str(hc),
        "cost_value": str(cost_value),
        "exec_time": str(exec_time),
    }
    chart_data = {
        "dataset_png": dataset_png,
        "algorithm_png": algorithm_png,
        "cost_function": cost_function,
        "l_param": l_param,
    }

    return table_data, chart_data


def get_dataset(dataset_name):
    dataset = pd.read_csv(DATASET_PATH + dataset_name + ".lrn", sep="\s+").drop("Key", axis=1)
    return dataset


def get_dataset_labels(dataset_name):
    labels = pd.read_csv(DATASET_LABELS_PATH + dataset_name + ".txt", sep="\s+").drop(
        "datapoint", axis=1
    )
    return labels.to_numpy().flatten()


def get_dataset_clusters(dataset_name):
    return DATASET_N_CLUSTERS[dataset_name]


def get_dataset_dimension(dataset_name):
    return DATASET_DIMENSION[dataset_name]
