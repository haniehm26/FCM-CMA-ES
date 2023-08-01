import time

import numpy as np
import pandas as pd

from sklearn.metrics import rand_score

from typing import Optional
from pydantic import BaseModel, Extra, Field, validate_arguments

from .plots import visualize_fcm_result, visualize_cost_function
from .constants import DATASET_N_CLUSTERS, DATASET_DIMENSION, DATASET_PATH, DATASET_LABELS_PATH


class FCM(BaseModel):
    # Step 1
    n_clusters: int = Field(5, ge=1, le=100)
    m: float = Field(2.0, ge=1.0)
    max_iter: int = Field(150, ge=1, le=1000)
    error: float = Field(1e-5, ge=1e-9)
    random_state: Optional[int] = None
    trained: bool = Field(False, Literal=True)

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
    n_clusters: int = Field(5, ge=1, le=100)
    m: float = Field(2.0, ge=1.0)
    random_state: Optional[int] = None
    trained: bool = Field(False, Literal=True)

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True

    @validate_arguments
    def fit(self, X):
        n_samples = X.shape[0]
        self.rng = np.random.default_rng(self.random_state)
        self.u, self._centers, J, j, L = FCM_CMA._cma(
            data=X, degree=self.m, domain=[np.min(X), np.max(X)], d=X.shape[1], n=self.n_clusters
        )
        self.u = self.soft_predict(X)
        self._centers = FCM_CMA._next_centers(X, self.u, self.m)
        self.trained = True
        return J, j, L

    def soft_predict(self, X):
        temp = FCM_CMA._dist(X, self._centers) ** float(2 / (self.m - 1))
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        temp = temp[:, :, np.newaxis]
        denominator_ = temp / denominator_
        denominator_ = denominator_.sum(2)
        return 1 / denominator_

    @staticmethod
    def _next_centers(X, u, m):
        um = u**m
        uX = X.T @ um
        return (uX / np.sum(um, axis=0)).T

    @validate_arguments
    def predict(self, X):
        if self.is_trained():
            X = np.expand_dims(X, axis=0) if len(X.shape) == 1 else X
            return self.soft_predict(X).argmax(axis=-1)

    def is_trained(self):
        if self.trained:
            return True
        raise ReferenceError("You need to train the model. Run `.fit()` method to this.")

    @staticmethod
    def _dist(A, B):
        A = A[:, None, :]
        dist = (A - B) ** 2
        return np.sqrt(np.einsum("ijk->ij", dist))

    @property
    def centers(self):
        if self.is_trained():
            return self._centers

    @property
    def partition_coefficient(self):
        if self.is_trained():
            return np.mean(self.u**2)

    @property
    def partition_entropy_coefficient(self):
        if self.is_trained():
            return -np.mean(self.u * np.log2(self.u))

    # ------------------- CMA PART -------------------
    class Chromosome:
        def __init__(self, n_samples, n_clusters, dimension):
            self.u = np.zeros((n_samples, n_clusters))
            self.v = np.zeros((n_clusters, dimension))

        def get_chromosome(self):
            return self.u, self.v

        def get_u(self):
            return self.u

        def get_v(self):
            return self.v

        def set_chromosome(self, u, v):
            self.u = u
            self.v = v

    @staticmethod
    def _euclidean_dist(A, B):
        A = A[:, None, :]
        B = B[:, None, :]
        dist = (A - B) ** 2
        return np.sqrt(np.einsum("ijkl->ijk", dist))

    @staticmethod
    def _cost_function(X, U, V, L, m):
        return (((U**m) * FCM_CMA._euclidean_dist(X, V)).sum(2)).sum(1) - (
            L * (U.sum(axis=2) - 1)
        ).sum()

    @staticmethod
    def _cma(data, degree, domain, d, n):
        # U = S × N, V = N × D
        s = data.shape[0]

        # domain
        V_d_min = np.min(domain)
        V_d_max = np.max(domain)
        U_d_min = 0
        U_d_max = 1

        # initialize population
        m = FCM_CMA.Chromosome(s, n, d)
        m.set_chromosome(
            u=np.random.uniform(low=U_d_min, high=U_d_max, size=(1, s, n)),
            v=np.random.uniform(low=V_d_min, high=V_d_max, size=(1, n, d)),
        )

        # sigma(U, V)
        sigma = [0.5, 0.3 * (V_d_max - V_d_min)]

        # termination parameters
        stop_fitness = 1e-1
        stop_generation = 1e3 * (d * n) ** 2

        # lambda & mu
        LAMBDA = int(4 + np.floor(3 * np.log(d)))
        MU = int(LAMBDA / 2)

        # weights
        w = np.log(MU + 1 / 2) - np.log(range(1, (MU) + 1)).reshape(MU, 1)
        w = w / np.sum(w)

        # step size control parameters
        mueff = 1 / np.sum(w**2)
        cs = (mueff + 2) / ((d) + mueff + 5)
        ds = 1 + 2 * np.maximum(0, np.sqrt(np.abs((mueff - 1) / ((d) + 1))) - 1) + cs

        # covariance matrix parameters
        alpha = 2
        cc = (4 + mueff / (d)) / ((d) + 4 + 2 * mueff / (d))
        c1 = alpha / (((d) + 1.3) ** 2 + mueff)
        cmu = np.minimum(
            1 - c1, alpha * ((mueff - 2 + 1 / mueff) / (((d) + 2) ** 2 + alpha * mueff / 2))
        )

        # evolution path
        ps = [np.zeros((n, 1)), np.zeros((d, 1))]
        pc = [np.zeros((n, 1)), np.zeros((d, 1))]

        # matrices
        update_matrix = 0
        B = [np.eye(n), np.eye(d)]
        D = [np.eye(n), np.eye(d)]
        C = [B[0] @ D[0] @ (B[0] @ D[0]).T, B[1] @ D[1] @ (B[1] @ D[1]).T]

        # expectation of ||N(0, I)||
        E = (d) ** 0.5 * (1 - 1 / (4 * (d)) + 1 / (21 * (d) ** 2))

        l = 0.001

        J = []
        L = []
        champion = []

        # loop
        population_size = 0
        while population_size < stop_generation:
            # generate offspring
            z = [
                np.random.randn(s * n * LAMBDA).reshape(n, -1),
                np.random.randn(d * n * LAMBDA).reshape(d, -1),
            ]
            v = [(B[0] @ D[0] @ z[0]).reshape(-1, s, n), (B[1] @ D[1] @ z[1]).reshape(-1, n, d)]

            # mutation
            x = [m.get_u() + sigma[0] * v[0], m.get_v() + sigma[1] * v[1]]

            # check offspring U domain
            while (x[0] > 1).any() or (x[0] < 0).any():
                z[0] = np.random.randn(s * n * LAMBDA).reshape(n, -1)
                v[0] = (B[0] @ D[0] @ z[0]).reshape(-1, s, n)
                where_min = np.where(x[0] < 0)
                where_max = np.where(x[0] > 1)
                x[0][where_min] = (m.get_u() + sigma[0] * v[0])[where_min]
                x[0][where_max] = (m.get_u() + sigma[0] * v[0])[where_max]
            #             x[0] = x[0] / np.repeat(x[0].sum(axis=2), n, axis=1).reshape(-1, s, n)

            # check offspring V domain
            while (x[1] > V_d_max).any() or (x[1] < V_d_min).any():
                z[1] = np.random.randn(d * n * LAMBDA).reshape(d, -1)
                v[1] = (B[1] @ D[1] @ z[1]).reshape(-1, n, d)
                where_min = np.where(x[1] < V_d_min)
                where_max = np.where(x[1] > V_d_max)
                x[1][where_min] = (m.get_v() + sigma[1] * v[1])[where_min]
                x[1][where_max] = (m.get_v() + sigma[1] * v[1])[where_max]

            # calculate fitness
            fitness = FCM_CMA._cost_function(X=data, U=x[0], V=x[1], L=l, m=degree)
            population_size += LAMBDA
            generation = population_size / LAMBDA

            # select parents by rank
            ranks = np.argsort(np.argsort(fitness.reshape(LAMBDA)))
            selected = [i for i, _ in enumerate(ranks) if ranks[i] < MU]

            # recombination
            m.set_chromosome(
                (x[0][selected].T @ w).reshape(-1, s, n), (x[1][selected].T @ w).reshape(-1, n, d)
            )
            z = [z[0].reshape(-1, s, n), z[1].reshape(-1, n, d)]
            y = [(z[0][selected].T @ w).reshape(n, -1), (z[1][selected].T @ w).reshape(d, -1)]

            # update evolution paths
            ps = [
                (1 - cs) * ps[0] + (np.sqrt(cs * (2 - cs) * mueff)) * (B[0] @ y[0]),
                (1 - cs) * ps[1] + (np.sqrt(cs * (2 - cs) * mueff)) * (B[1] @ y[1]),
            ]
            hs = [
                np.minimum(
                    np.linalg.norm(ps[0])
                    / np.sqrt(1 - (1 - cs) ** (2 * population_size / LAMBDA))
                    / E,
                    1.4 + 2 / ((d * n) + 1),
                ),
                np.minimum(
                    np.linalg.norm(ps[1])
                    / np.sqrt(1 - (1 - cs) ** (2 * population_size / LAMBDA))
                    / E,
                    1.4 + 2 / ((d * n) + 1),
                ),
            ]
            pc = [
                (1 - cc) * pc[0] + hs[0] * np.sqrt(cc * (2 - cc) * mueff) * (B[0] @ D[0] @ y[0]),
                (1 - cc) * pc[1] + hs[1] * np.sqrt(cc * (2 - cc) * mueff) * (B[1] @ D[1] @ y[1]),
            ]

            # update covariance matrix C
            M = [
                B[0] @ D[0] @ z[0][selected].reshape(n, -1),
                B[1] @ D[1] @ z[1][selected].reshape(d, -1),
            ]
            C = [
                (1 - c1 - cmu) * C[0]
                + c1 * (pc[0] @ pc[0].T + (1 - hs[0]) * cc * (2 - cc) * C[0])
                + cmu * M[0] * np.diag(w) @ M[0].T,
                (1 - c1 - cmu) * C[1]
                + c1 * (pc[1] @ pc[1].T + (1 - hs[1]) * cc * (2 - cc) * C[1])
                + cmu * M[1] * np.diag(w) @ M[1].T,
            ]

            # update step-size
            sigma = [
                sigma[0] * np.exp((cs / ds) * (np.linalg.norm(ps[0]) / E - 1)),
                sigma[1] * np.exp((cs / ds) * (np.linalg.norm(ps[1]) / E - 1)),
            ]

            # update B and D from C
            cone = 0
            if population_size - update_matrix > LAMBDA / (cone + cmu) / (d) / 10:
                update_matrix = population_size
                C = [np.triu(C[0]) + np.triu(C[0], 1).T, np.triu(C[1]) + np.triu(C[1], 1).T]
                B[0], D[0] = np.linalg.eig(C[0])
                B[1], D[1] = np.linalg.eig(C[1])
                B = [B[0] * np.eye(n), B[1] * np.eye(d)]
                D = [
                    np.diag(np.sqrt(np.abs(np.diag(D[0]).reshape(-1, 1)))) * np.eye(n),
                    np.diag(np.sqrt(np.abs(np.diag(D[1]).reshape(-1, 1)))) * np.eye(d),
                ]

            best_J = np.min(fitness[selected])
            i = np.where(best_J == fitness[ranks])

            L.append(l)

            #             update parameter l
            l = 1.001 * l
            #             if generation%10!=0:
            #                 champion.append(x[0][ranks[i]])
            #             if generation%10==0:
            #                 feasible = False
            #                 for j in range(len(champion)):
            #                     if (champion[j].sum(2)-1).sum()==0:
            #                         feasible = True
            #                 if feasible:
            #                     l = l*0.75
            #                 else:
            #                     l = l/0.75
            #                 champion = []

            J.append(best_J)

            if best_J <= stop_fitness:
                break

            if best_J == fitness[int(np.ceil(0.7 * LAMBDA))]:
                sigma = [sigma[0] * np.exp(0.2 + cs / ds), sigma[1] * np.exp(0.2 + cs / ds)]

            # check sigma U bound
            if sigma[0] > U_d_max:
                B[0] = np.eye(n)
                D[0] = np.eye(n)
                C[0] = B[0] @ D[0] @ (B[0] @ D[0]).T
                ps[0] = np.zeros((n, 1))
                pc[0] = np.zeros((n, 1))
                sigma[0] = 0.5

            # check sigma V bound
            if sigma[1] > V_d_max:
                B[1] = np.eye(d)
                D[1] = np.eye(d)
                C[1] = B[1] @ D[1] @ (B[1] @ D[1]).T
                ps[1] = np.zeros((d, 1))
                pc[1] = np.zeros((d, 1))
                sigma[1] = 0.3 * (V_d_max - V_d_min)

        x[0] = x[0] / np.repeat(x[0].sum(axis=2), n, axis=1).reshape(-1, s, n)
        return x[0][ranks[i[0][0]]].reshape(s, n), x[1][ranks[i[0][0]]].reshape(n, d), J, best_J, L


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


def call_fcm_cma(m, l, dataset_name):
    dataset = get_dataset(dataset_name)
    dataset_labels = get_dataset_labels(dataset_name)
    dataset_clusters = get_dataset_clusters(dataset_name)
    dataset_dimension = get_dataset_dimension(dataset_name)

    start_time = time.time()
    # need to add l parameter
    fcm_cma = FCM_CMA(n_clusters=dataset_clusters, m=m, error=1e-8)
    J, cost_value, L = fcm_cma.fit(dataset.to_numpy())
    cost_value = format(cost_value, ".2f")
    fcm_cma_labels = fcm_cma.predict(dataset.to_numpy())
    exec_time = format(time.time() - start_time, ".4f")

    fc = format(fcm_cma.partition_coefficient, ".2f")
    hc = format(fcm_cma.partition_entropy_coefficient, ".2f")

    rand_index = format(rand_score(dataset_labels, fcm_cma_labels), ".2f")

    dataset_png = visualize_fcm_result(
        dataset, dataset_clusters, dataset_labels, dataset_dimension, True
    )

    algorithm_png = visualize_fcm_result(
        dataset, dataset_clusters, fcm_cma_labels, dataset_dimension, False
    )

    cost_function = visualize_cost_function(J, cost_value)

    chart_data = {
        "dataset_png": dataset_png,
        "cost_function": cost_function,
        "algorithm_png": algorithm_png,
    }

    if l != "zero":
        chart_data["l_param"] = visualize_cost_function(L, 0)

    table_data = {
        "rand_index": str(rand_index),
        "fc": str(fc),
        "hc": str(hc),
        "cost_value": str(cost_value),
        "exec_time": str(exec_time),
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
