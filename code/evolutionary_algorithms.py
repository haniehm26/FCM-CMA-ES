import time

import numpy as np

from .constants import DOMAIN
from .plots import contour_plot, es_plot, sigma_plot


def es(
    domain,
    objective_function,
    keep_parent,
    n_generation=5000,
    SIGMA=0.15,
    LAMBDA=500,
    MU=50,
    DIM=2,
):
    start_time = time.time()

    X, y = None, np.inf

    x_total, y_total = [], []
    x_best, y_best = [], []

    # termination
    stop_fitness = 1e-8

    # number of offsprings per parent
    n_offspring = int(LAMBDA / MU) if LAMBDA >= MU else 1

    selcetion_type = "(µ + ʎ)" if keep_parent else "(µ, ʎ)"

    # population range
    d_min = np.min(domain)
    d_max = np.max(domain)

    # initialize population
    population = np.random.uniform(low=d_min, high=d_max, size=(LAMBDA, DIM)).reshape(
        LAMBDA, -1, DIM
    )

    for _ in range(n_generation):
        # calculate fitness for each individual in the population
        fitness = objective_function(population)

        # the less fitness, the higher rank
        ranks = np.argsort(np.argsort(fitness.reshape(LAMBDA)))

        # indices of first MU ranks
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < MU]

        next_generation_population = []

        for i in selected:
            x_total.append(population[i, :, :])
            y_total.append(fitness[i])

            # check if this parent is the best solution ever seen
            if fitness[i] < y:
                X, y = population[i, :, :], fitness[i]
                x_best.append(X)
                y_best.append(y)

            if keep_parent:
                next_generation_population.append(population[i, :, :])

            # create offsprings of parents
            for _ in range(n_offspring):
                offspring = np.inf
                while offspring is np.inf or offspring.any() > d_max or offspring.any() < d_min:
                    # mutation
                    offspring = (population[i, :, :] + np.random.randn(DIM) * SIGMA).reshape(
                        -1, DIM
                    )
                next_generation_population.append(offspring)

        if keep_parent:
            LAMBDA = len(next_generation_population)

        # replace current generation with the next generation population
        population = np.array(next_generation_population).reshape(LAMBDA, -1, DIM)

        # terminate if fitness is good enough
        if y <= stop_fitness:
            print("fitness is good enough")
            break

    exec_time = format(time.time() - start_time, ".4f")

    if DIM == 2:
        total_title = (
            "Total Samples of "
            + objective_function.__name__
            + ", Selection "
            + selcetion_type
            + ", D="
            + str(DIM)
        )
        best_title = (
            "Best Samples of "
            + objective_function.__name__
            + ", Selection "
            + selcetion_type
            + ", D="
            + str(DIM)
        )

        es_total = es_plot(np.asarray(x_total), np.asarray(y_total), total_title)
        es_best = es_plot(np.asarray(x_best), np.asarray(y_best), best_title)

        domain = DOMAIN[objective_function.__name__.lower()]

        es_total_contour = contour_plot(
            domain, 15, objective_function, np.asarray(x_total), total_title
        )
        es_best_contour = contour_plot(
            domain, 15, objective_function, np.asarray(x_best), best_title
        )

        chart_data = {
            "es_total": str(es_total),
            "es_best": str(es_best),
            "es_total_contour": str(es_total_contour),
            "es_best_contour": str(es_best_contour),
        }

    else:
        chart_data = str(None)

    table_data = {"X": str(X), "y": str(y), "exec_time": str(exec_time)}

    return table_data, chart_data


def cma(domain, objective_function, dim):
    start_time = time.time()

    X = []
    Sigma = []
    result_X, result_Y = 0, 0

    # objective function domain
    d_min = np.min(domain)
    d_max = np.max(domain)

    # initialize population
    m = np.random.uniform(low=d_min, high=d_max, size=(dim, 1))
    sigma = 0.3 * (d_max - d_min)

    # termination parameters
    stop_fitness = 1e-10
    stop_generation = (1e3 * dim**2) * 10

    # lambda & mu
    LAMBDA = int(4 + np.floor(3 * np.log(dim)))
    MU = int(LAMBDA / 2)

    # weights
    w = np.log(MU + 1 / 2) - np.log(range(1, MU + 1)).reshape(MU, 1)
    w = w / np.sum(w)

    # step size control parameters
    mueff = 1 / np.sum(w**2)
    cs = (mueff + 2) / (dim + mueff + 5)
    ds = 1 + 2 * np.maximum(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

    # covariance matrix parameters
    alpha = 2
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    c1 = alpha / ((dim + 1.3) ** 2 + mueff)
    cmu = np.minimum(
        1 - c1, alpha * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + alpha * mueff / 2))
    )

    # evolution path
    ps = np.zeros((dim, 1))
    pc = np.zeros((dim, 1))

    # matrices
    B = np.eye(dim)
    D = np.eye(dim)
    C = B @ D @ (B @ D).T
    update_matrix = 0

    # expectation of ||N(0, I)||
    E = dim**0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

    # loop
    population_size = 0
    while population_size < stop_generation:
        # generate offspring
        z = np.random.randn(dim * LAMBDA).reshape(dim, LAMBDA)
        v = B @ D @ z

        # mutation
        x = m + sigma * v

        # check offspring domain
        while (x > d_max).any() or (x < d_min).any():
            z = np.random.randn(dim * LAMBDA).reshape(dim, LAMBDA)
            v = B @ D @ z
            where_min = np.where(x < d_min)
            where_max = np.where(x > d_max)
            x[where_min] = (m + sigma * v)[where_min]
            x[where_max] = (m + sigma * v)[where_max]

        # calculate fitness
        fitness = objective_function(x.reshape(LAMBDA, -1, dim))
        population_size += LAMBDA

        # select parents by rank
        ranks = np.argsort(np.argsort(fitness.reshape(LAMBDA)))
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < MU]

        x = x.reshape(LAMBDA, -1)
        z = z.reshape(LAMBDA, -1)
        y = np.zeros((dim, 1))

        # recombination
        m = x[selected].reshape(dim, -1) @ w
        y = z[selected].reshape(dim, -1) @ w

        # update evolution paths
        ps = (1 - cs) * ps + (np.sqrt(cs * (2 - cs) * mueff)) * (B @ y)
        hs = np.minimum(
            np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * population_size / LAMBDA)) / E,
            1.4 + 2 / (dim + 1),
        )
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * (B @ D @ y)

        # update covariance matrix C
        M = B @ D @ z[selected].reshape(dim, -1)
        C = (
            (1 - c1 - cmu) * C
            + c1 * (pc @ pc.T + (1 - hs) * cc * (2 - cc) * C)
            + cmu * M * np.diag(w) @ M.T
        )

        # update step-size
        Sigma.append(sigma)
        sigma = sigma * np.exp((cs / ds) * (np.linalg.norm(ps) / E - 1))

        # update B and D from C
        cone = 0
        if population_size - update_matrix > LAMBDA / (cone + cmu) / dim / 10:
            update_matrix = population_size
            C = np.triu(C) + np.triu(C, 1).T
            B, D = np.linalg.eig(C)
            B = B * np.eye(dim)
            D = np.diag(np.sqrt(np.abs(np.diag(D).reshape(-1, 1)))) * np.eye(dim)

        if fitness[ranks[0]] <= stop_fitness:
            print("fitness is good enough")
            break

        if fitness[ranks[0]] == fitness[int(np.ceil(0.7 * LAMBDA))]:
            sigma = sigma * np.exp(0.2 + cs / ds)
            # print("local minimum risk")
            X.append(x)

        # check sigma bound
        if sigma > 10 * (d_max - d_min):
            # print("update sigma")
            B = np.eye(dim)
            D = np.eye(dim)
            C = B @ D @ (B @ D).T
            ps = np.zeros((dim, 1))
            pc = np.zeros((dim, 1))
            sigma = 0.3 * (d_max - d_min)

        X.append(x)

        result_X = np.real(x[ranks[0]])
        result_Y = np.real(fitness[ranks[0]])

    n_generation = int(population_size / LAMBDA)
    exec_time = format(time.time() - start_time, ".4f")

    if dim == 2:
        title = "Samples of " + objective_function.__name__ + ", D=" + str(dim)

        domain = DOMAIN[objective_function.__name__.lower()]

        cma_contour = contour_plot(domain, 15, objective_function, np.asarray(X), title)
        sigma_chart = sigma_plot(Sigma, n_generation, dim, objective_function)

        chart_data = {
            "cma_contour": str(cma_contour),
            "sigma_chart": str(sigma_chart),
        }

    else:
        chart_data = str(None)

    table_data = {
        "n_generation": str(n_generation),
        "n_offspring": str(LAMBDA),
        "n_parent": str(MU),
        "X": str(result_X),
        "y": str(result_Y),
        "exec_time": str(exec_time),
    }

    return table_data, chart_data
