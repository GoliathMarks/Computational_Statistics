from HomeworkFour.CompStatsHomeworkFourLib import CompStatsHomeworkFourLib as cs4
import numpy as np

from scipy.stats import t
from typing import List, Tuple


class InfectionRates:
    class CountryData:
        def __init__(self, rate_before, rate_after):
            self.rate_before = rate_before
            self.rate_after = rate_after

    def __init__(self, filename):
        self.countries_data = {}
        self.t_statistic = None
        self.bootstrap_statistic = None
        self.reject_t_null_hypothesis = False
        self.reject_bootstrap_null_hypothesis = False
        self.dof = 0

        self.initialize_from_file(filename=filename)
        self.perform_t_test()

    def initialize_from_file(self, filename):
        with open(filename, "r") as f:
            lines: List[Tuple] = [
                (
                    float(line.strip("\n").split(",")[0]),
                    float(line.strip("\n").split(",")[1]),
                    line.strip("\n").split(",")[2]
                 ) for line in f.readlines()[1:]]

        for line in lines:
            self.countries_data[line[2]] = self.CountryData(rate_before=line[0], rate_after=line[1])
            self.dof = len(self.countries_data)

    def perform_t_test(self):
        xs: np.ndarray = np.array([v.rate_before for k, v in self.countries_data.items()])
        ys: np.ndarray = np.array([v.rate_after for k, v in self.countries_data.items()])

        diffs: np.ndarray = xs - ys
        delta_bar: float = diffs.mean()
        sigma_delta: float = np.sqrt(diffs.var(ddof=1))
        self.t_statistic = np.sqrt(diffs.size) * (delta_bar / sigma_delta)

    def decide_null_t_hypothesis(self, alpha):
        endpints: Tuple = t.interval(alpha, self.dof)
        print(f"endpoints = {endpints}")
        if self.t_statistic < endpints[0] or self.t_statistic > endpints[1]:
            self.reject_t_null_hypothesis = True

    def perform_bootstrap_test(self, num_bootstrap_repetitions):
        xs: np.ndarray = np.array([v.rate_before for k, v in self.countries_data.items()])
        ys: np.ndarray = np.array([v.rate_after for k, v in self.countries_data.items()])
        diffs: np.ndarray = xs - ys
        t_obs: float = np.abs(xs.mean() - ys.mean())/(diffs.var(ddof=1))
        z_star: np.ndarray = np.concatenate((xs, ys), axis=0)
        t_star_statistics: List = []

        for i in range(num_bootstrap_repetitions):
            x_star_indices: np.ndarray = np.random.randint(0, 32, size=16)
            # print(f"x_star_indices = {x_star_indices}")
            y_star_indices: np.ndarray = np.random.randint(0, 32, size=16)
            # print(f"y_star_indices = {y_star_indices}")
            x_star: np.ndarray = np.array([z_star[j] for j in x_star_indices])
            # print(f"x_star = {x_star}")
            y_star: np.ndarray = np.array([z_star[j] for j in y_star_indices])
            # print(f"y_star= {y_star}")
            diff_star: np.ndarray = x_star - y_star
            # print(f"diff_star = {diff_star}")
            t_star: np.ndarray = np.abs(x_star.mean() - y_star.mean())/(diff_star.var(ddof=1))
            # print(f"t_star = {t_star}")
            t_star_statistics.append(t_star)

        t_star_statistics: np.ndarray = np.array(t_star_statistics)
        print(f"t observed = {t_obs}")
        print(f"t_star_statistics = {t_star_statistics}")
        bigga_ts: np.ndarray = np.array([stat for stat in t_star_statistics.tolist() if stat >= t_obs])
        probability_estimate: float = bigga_ts.size/num_bootstrap_repetitions
        self.bootstrap_statistic = probability_estimate
        print(f"Bootstrap probability estimte: {probability_estimate}")

    def decide_null_bootstrap_hypothesis(self, alpha):
        if self.bootstrap_statistic <+ alpha:
            self.reject_bootstrap_null_hypothesis = True
        else:
            self.reject_bootstrap_null_hypothesis = False




date_file = "/home/ryan/PycharmProjects/ComputationalStatistics/CompStatsHomeworkFive/data/ex5task1.csv"
data: InfectionRates = InfectionRates(filename=date_file)

data.perform_t_test()
print(data.t_statistic)
data.decide_null_t_hypothesis(0.95)
print(data.reject_t_null_hypothesis)

data.perform_bootstrap_test(10)
data.decide_null_bootstrap_hypothesis(.05)
print(data.reject_bootstrap_null_hypothesis)
