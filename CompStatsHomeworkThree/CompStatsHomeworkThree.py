"""Computational Statistics Exercise 3
Author: Ryan Hutchins
"""
import matplotlib.pyplot as plt
import numpy as np
import codecs

from datetime import datetime
from numpy.random import rand
from numpy.random import seed
from typing import List, Tuple

"""CODE FOR PROBLEM 1 STARTS HERE"""


class SampleGenerator:
    def __init__(self, sample_size_range, number_of_samples, parameter):
        self.sample_size_range: range = sample_size_range
        self.number_of_samples: int = number_of_samples
        self.parameter: float = parameter
        self.current_sample: List[np.ndarray] = None

    def generate_sample(self, size_of_each_sample: int) -> np.ndarray:
        """Creates a kxn numpy array drawn where the rows are records and the the columns are samples in that record"""
        data = np.random.poisson(self.parameter, (self.number_of_samples, size_of_each_sample))
        return data

    def generate_samples(self) -> List[np.ndarray]:
        data: List[np.ndarray] = [self.generate_sample(sample_size) for sample_size in self.sample_size_range]
        self.current_sample = data
        return data

    def compute_mean_sample_standard_deviation(self, record_index: int) -> float:
        standard_deviations = [np.sqrt(sample.var()) for sample in self.current_sample[record_index]]
        ye_ole_runnin_tally = 0  # Variables named with whimsy
        for std in standard_deviations:
            ye_ole_runnin_tally += std
        mean_sample_std_deviation = ye_ole_runnin_tally / self.number_of_samples
        return mean_sample_std_deviation

    def compute_mean_sample_standard_deviations(self) -> List[float]:
        return [self.compute_mean_sample_standard_deviation(i) for i in range(len(self.current_sample))]

    def compute_sample_standard_error(self, record_index: int) -> float:
        standard_errors_of_samples = [np.power(sample.mean()-self.parameter, 2) for sample in self.current_sample[record_index]]
        tally_ho = 0
        for m in standard_errors_of_samples:
            tally_ho += m
        avg_mean = tally_ho / self.number_of_samples
        return avg_mean

    def compute_sample_standard_errors(self) -> List[float]:
        return [self.compute_sample_standard_error(j) for j in range(len(self.current_sample))]

    def plot_mean_sample_std_dev_vs_sample_size(self):
        x = np.array([j for j in self.sample_size_range])
        y = np.array(self.compute_mean_sample_standard_deviations())
        y2 = self.parameter*((x-1)/x)
        bias = y - self.parameter

        plt.scatter(x, y, c='g', alpha=0.5, label="mean sample std. dev.")
        plt.plot(x, y2, c='k', label='sigma*((N-1)/N)')
        plt.plot(x, bias, c='r', label=' sample bias')
        plt.title('mean sample standard deviation vs sample size')
        plt.xlabel('sample size')
        plt.xticks([10, 20, 30, 40, 50])
        plt.ylabel('mean sample st dev')
        plt.legend()
        plt.show()

    def plot_standard_sample_mean_error_against_n(self):
        x = np.array([j for j in self.sample_size_range])
        y = np.array(self.compute_sample_standard_errors())

        plt.scatter(x, y, c='g', alpha=0.5, label="standard error vs sample size")
        plt.title('mean sample standard deviation vs sample size')
        plt.xlabel('sample size')
        plt.xticks([10, 20, 30, 40, 50])
        plt.ylabel('mean sample st dev')
        plt.legend()
        plt.show()


def do_problem_one(sample_size_range: range, number_of_samples: int, parameter: float):
    generator = SampleGenerator(sample_size_range=sample_size_range, number_of_samples=number_of_samples, parameter=parameter)
    generator.generate_samples()
    generator.plot_mean_sample_std_dev_vs_sample_size()
    generator.plot_standard_sample_mean_error_against_n()


"""CODE FOR PROBLEM 2 STARTS HERE"""


def strip_bom(filename, new_filename):
    with open(filename, "rb") as f:
        lines = f.read()
        lines = lines[len(codecs.BOM_UTF8):]
        with open(new_filename, "wb") as f2:
            f2.write(lines)


def read_data(filename) -> List[float]:
    with open(filename, "r", newline='') as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            new_lines.append(int(line.replace("\r\n", "")))
    return new_lines


def get_cumulative_sum(data: List[float]) -> np.ndarray:
    cum_sum_array: np.ndarray = np.zeros(len(data))
    total = 0
    for i, entry in enumerate(data):
        total += entry
        cum_sum_array[i] = total

    return cum_sum_array


def zip_output_variables_with_predictors(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_vals: np.ndarray = np.array(list(range(len(data))))
    return x_vals, data


def plot_cumulative_sum_and_linear_mle_model(cum_sum):
    """
        Plot the cumulative sum of the cases against the day number since the start of the records along with the
        best-fit line according to the MLE.

        Then, in a separate graph, plot the residuals.
    """
    data = zip_output_variables_with_predictors(cum_sum)
    x = data[0]
    y = data[1]

    beta_1_hat = compute_beta_1_hat(data=data)
    beta_0_hat = compute_beta_0_hat(data=data, beta_1_hat=beta_1_hat)

    y2 = beta_0_hat + (beta_1_hat * x)

    residuals_vector = y - y2

    fig, axs = plt.subplots(2)

    axs[0].scatter(x, y, c='g', alpha=0.5, label="cum cases vs day number")
    axs[0].plot(x, y2, c='r', label="MLE best fit line")
    axs[0].set(
        title="cumulative new infections of COVID-19",
        xlabel="day number",
        xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        ylabel="total cases diagnosed"
    )
    axs[0].legend()

    axs[1].scatter(x, residuals_vector, c='k', label="residuals")
    axs[1].set(
        title="residual vs day number",
        xlabel="day number",
        ylabel="residual value",
        xticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    )
    axs[1].legend()

    plt.show()


def compute_beta_1_hat(data: Tuple[np.ndarray, np.ndarray]) -> float:
    """Compute beta_1 parameter, which is done exclusively from the data. For MLE in this case, we have the closed
    form solution:

        beta_1_hat = c(x,y)/(s_x)^2

    Use the closed form solution for the parameter derived in class."""
    x_vals = data[0]
    y_vals = data[1]
    cxy = np.cov(x_vals, y_vals)[0][1]
    sx2 = x_vals.var()
    beta_1 = cxy/sx2
    return beta_1


def compute_beta_0_hat(data: Tuple[np.ndarray, np.ndarray], beta_1_hat: float):
    """
        Computes the optimal parameter beta_1, which in MLE has the closed-form solution:

            beta_0_hat = y_bar - (beta_1_hat * x_bar)

    """
    x = data[0]
    x_bar = x.mean()
    y = data[1]
    y_bar = y.mean()
    beta_0 = y_bar - (beta_1_hat * x_bar)
    return beta_0


def do_problem_two():
    d: List[float] = read_data(
        "/Users/administrator/PycharmProjects/ComputationalStatistics/CompStatsHomeworkThree/data/covid19data.csv"
    )
    cs: np.ndarray = get_cumulative_sum(d)
    plot_cumulative_sum_and_linear_mle_model(cs)

"""CODE FOR PROBLEM 3 STARTS HERE"""
def compute_probability(x: float, beta: int=3, theta: int=1) -> np.ndarray:
    return 1/(1 + np.power(np.e, beta * (theta - x)))


def generate_data_set(beta: int=3, theta: int=1, size: int=10000) -> Tuple[np.ndarray, np.ndarray]:
    xs: np.ndarray = -10 + np.random.rand(size) * 20
    xs_list: List[float] = xs.tolist()
    ys = [np.random.binomial(n=1, p=compute_probability(x, beta=beta, theta=theta)) for x in xs_list]
    return xs, np.array(ys)


"""No further time to work on problem set."""
def fix_beta_and_plot_across_theta():
    pass


def fix_theta_and_plot_across_beta():
    pass


def do_problem_three_part_b():
    pass


do_problem_one(sample_size_range=range(2,51), number_of_samples=1000, parameter=0.2)
do_problem_two()
print(generate_data_set(beta=3, theta=1, size=10))
