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
        self.reject_null_hypothesis = False
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
            self.reject_null_hypothesis = True


date_file = "/home/ryan/PycharmProjects/ComputationalStatistics/CompStatsHomeworkFive/data/ex5task1.csv"
data: InfectionRates = InfectionRates(filename=date_file)

data.perform_t_test()
print(data.t_statistic)
data.decide_null_t_hypothesis(0.95)
print(data.reject_null_hypothesis)
