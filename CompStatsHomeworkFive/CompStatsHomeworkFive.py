from HomeworkFour.CompStatsHomeworkFourLib import CompStatsHomeworkFourLib as cs4
import numpy as np

from scipy.stats import t, chi2
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


class WeatherData:
    def __init__(self, filename):
        self.data = {}
        self.reject_likelihood_ratio_hypothesis = False
        self.chi_squared_probability = -1
        self.initialize_from_file(filename)
        ones: np.ndarray = np.ones(360)
        xs: np.ndarray = np.array([k for k, v in self.data.items()])
        cosines: np.ndarray = np.cos((np.pi * xs) / 6.)
        self.y = np.array([v for k, v in self.data.items()])
        self.full_A = np.column_stack((ones, xs, cosines))
        full_model_output = self.obtain_least_squares_estimates(self.full_A)
        self.full_beta = full_model_output[0]
        self.full_residuals_sum = full_model_output[1]
        self.reduced_A = np.column_stack((ones, cosines))
        reduced_model_output = self.obtain_least_squares_estimates(self.reduced_A)
        self.reduced_beta = reduced_model_output[0]
        self.reduced_residuals_sum = reduced_model_output[1]
        self.sample_variance = (1/(self.y.size - 1)) * (np.power(self.y - self.y.mean(), 2)).sum()
        self.D = -(1/self.sample_variance) * (self.full_residuals_sum - self.reduced_residuals_sum)

    def initialize_from_file(self, filename):
        with open(filename, "r") as f:
            lines: List = f.readlines()
            for line in lines:
                l: List = line.strip("\n").split(",")
                month_index = int(l[0])
                monthly_avg_temp = float(l[1])
                self.data[month_index] = monthly_avg_temp

    def obtain_least_squares_estimates(self, coefficients_matrix: np.ndarray):
        return np.linalg.lstsq(a=coefficients_matrix, b=self.y)

    def decide_chi_squared_null_hypothesis(self, alpha):
        endpoints = chi2.interval(alpha, 2)
        print(f"endpoints of the 95% chi_squared_interval: {endpoints}, D = {self.D}")
        if self.D < endpoints[0] or self.D > endpoints[1]:
            self.reject_likelihood_ratio_hypothesis = True
        else:
            self.reject_likelihood_ratio_hypothesis = False


date_file = "/home/ryan/PycharmProjects/ComputationalStatistics/CompStatsHomeworkFive/data/ex5task1.csv"
data: InfectionRates = InfectionRates(filename=date_file)

data.perform_t_test()
print(f"t_statistics = {data.t_statistic}")
data.decide_null_t_hypothesis(0.95)
print(f"Reject null hypothesis in t test boolean value: {data.reject_t_null_hypothesis}")

data.perform_bootstrap_test(1000)
data.decide_null_bootstrap_hypothesis(.05)
print(f"Reject null hypothesis in bootstrap test boolean value: {data.reject_bootstrap_null_hypothesis}")

data_file_2 = "/home/ryan/PycharmProjects/ComputationalStatistics/CompStatsHomeworkFive/data/ex5task2.csv"
w_data = WeatherData(filename=data_file_2)
print(f"Full beta parameters estimate: {w_data.full_beta}")
print(f"Reduced beta parameters estimate: {w_data.reduced_beta}")
w_data.decide_chi_squared_null_hypothesis(0.95)
print(w_data.reject_likelihood_ratio_hypothesis)
