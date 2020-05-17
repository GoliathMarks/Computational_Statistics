import matplotlib.pyplot as plt
import numpy as np
import os, sys, codecs

from typing import List


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


def strip_bom(filename, new_filename):
    with open(filename, "rb") as f:
        lines = f.read()
        lines = lines[len(codecs.BOM_UTF8):]
        with open(new_filename, "wb") as f2:
            f2.write(lines)


def read_data(filename):
    with open(filename, "r", newline='') as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            new_lines.append(int(line.replace("\r\n", "")))
    print(new_lines)
    return new_lines


def get_cumulative_sum(data) -> np.ndarray:
    cum_sum_array: np.ndarray = np.zeros(len(data))
    print(cum_sum_array)
    total = 0
    for i, entry in enumerate(data):
        total += entry
        cum_sum_array[i] = total

    return cum_sum_array


def plot_cumulative_sum(cum_sum):
    x = np.array([j for j in range(len(cum_sum))])
    y = cum_sum

    plt.scatter(x, y, c='g', alpha=0.5, label="standard error vs sample size")
    plt.title('cumulative new infections of COVID-19')
    plt.xlabel('day number')
    plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
    plt.ylabel('total cases diagnosed')
    plt.legend()
    plt.show()

def do_problem_two():
    pass


#  do_problem_one(sample_size_range=range(2,51), number_of_samples=1000, parameter=0.2)

d = read_data("/Users/administrator/PycharmProjects/ComputationalStatistics/CompStatsHomeworkThree/data/covid19data.csv")
print(get_cumulative_sum(d))
cs = get_cumulative_sum(d)
plot_cumulative_sum(cs)