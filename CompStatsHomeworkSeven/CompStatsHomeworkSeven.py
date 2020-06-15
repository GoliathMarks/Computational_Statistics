import numpy as np
import pandas
import matplotlib.pyplot as plt

from numpy.linalg import inv
from typing import List
from typing import Tuple


def get_data(filename) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(filename, "r") as f:
        lines: List[str] = f.readlines()[1:]
        xs: np.ndarray = np.array([float(line.split(",")[0]) for line in lines])
        standardized_xs = (xs - xs.mean()) / xs.std()
        ys: np.ndarray = np.array([float(line.split(",")[1].strip()) for line in lines])
        return xs, ys, standardized_xs


def compute_basis_polynomial_matrix(degree: int, x_values: np.ndarray) -> np.ndarray:
    """Computes the N x (degree + 1) matrix, called G in equation 2.38 from the script. In this case, the basis
    functions are of the form x^k."""
    records = []
    for value in x_values:
        record = [np.power(value, j) for j in range(0, degree+1)]
        records.append(np.array(record))
    return np.array(records)


def compute_mle(g: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.matmul(np.matmul(inv(np.matmul(g.transpose(), g)), g.transpose()), ys)


def make_predictions(g: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.matmul(g, beta)


def get_mean_squared_error(data: Tuple[np.ndarray, np.ndarray], predicted_values: np.ndarray):
    actual_ys: np.ndarray = data[1]
    differences: np.ndarray = actual_ys - predicted_values
    squared_errors: np.ndarray = np.power(differences, 2)
    sum_squared_errors: np.ndarray = squared_errors.sum()
    return (1 / squared_errors.size) * sum_squared_errors


def do_part_a(filename: str, degree: int):
    data: Tuple[np.ndarray, np.ndarray, np.ndarray] = get_data(filename=filename)
    G: np.ndarray = compute_basis_polynomial_matrix(degree=degree, x_values=data[0])
    beta: np.ndarray = compute_mle(G, data[1])
    predictions = make_predictions(g=G, beta=beta)
    mse = get_mean_squared_error(data=data, predicted_values=predictions)
    print(f"The mean squared error from our 21st-degree polynomial in part a is: {mse}")
    print(f"Plotting true vs. predicted data")
    print(data[0])
    print(data[1])
    plt.scatter(data[0], data[1], color='black', label="true")
    plt.plot(data[0], predictions, color="r", label="predicted")
    plt.legend()
    plt.show()





file = "/Users/administrator/PycharmProjects/ComputationalStatistics/CompStatsHomeworkSeven/data/sheet7.csv"
do_part_a(filename=file, degree=21)
