"""
    Computational Statistics Homework 4
    Author: Ryan Hutchins
    University of Heidelberg, Summer Somester 2020

    Serial Experiments in Social Isolation, day number: 12x, feels like day number: 4574573498

    Evey: Who are you?
    V: Who? Who is but the form following the function of what, and what I am is a man in a mask.
    Evey: Well, I can see that.
    V: Of course you can. I”m not questioning your powers of observation, I”m merely remarking upon the paradox of
        asking a masked man who he is.
"""
import matplotlib.pyplot as plt
import numpy as np

from scipy.special import comb
from typing import Tuple


"""Library functions pertaining to Problem 1, part a begin here."""


def get_xy_values(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a csv file comprising a set of x and y values"""
    with open(filename) as f:
        lines = f.readlines()
        lines = lines[1:]
        xs = [float(data.split(",")[0]) for data in lines]
        ys = [float(data.split(",")[1].strip("\n")) for data in lines]
    return np.array(xs), np.array(ys)


def compute_gradient(data: Tuple[np.ndarray, np.ndarray], parameters: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the likelihood function.
    """
    xs = data[0]
    ys = data[1]
    #  print(f"beta = {parameters[0]}, theta = {parameters[1]}")
    theta_minus_x_i: np.ndarray = parameters[1] - xs
    minus_beta_times_theta_minus_x_i: np.ndarray = -1 * parameters[0] * theta_minus_x_i
    exponentiated_theta_minus_x_i: np.ndarray = np.exp(minus_beta_times_theta_minus_x_i)
    exp_ratio: np.ndarray = exponentiated_theta_minus_x_i /(1 + exponentiated_theta_minus_x_i)
    exp_ratio_minus_y: np.ndarray = exp_ratio - ys
    first_component_vec: np.ndarray = exp_ratio_minus_y * theta_minus_x_i
    second_component_vec: np.ndarray = exp_ratio_minus_y * parameters[0]
    first_component = first_component_vec.sum()
    second_component = second_component_vec.sum()
    gradient = np.array([first_component, second_component])
    #  print(f"gradient = {gradient}")
    return gradient


def compute_log_likelihood_function_value(data: Tuple[np.ndarray, np.ndarray], parameters: np.ndarray) -> float:
    """
    Evaluates the value of the likelihood function on the dataset with the given values for beta and theta.
    """
    xs: np.ndarray = data[0]
    ys: np.ndarray = data[1]
    beta = parameters[0]
    theta = parameters[1]
    beta_times_theta_x_i: np.ndarray = beta * (theta - xs)
    likelihood_function_terms: np.ndarray = -ys * beta_times_theta_x_i - np.log(1 + np.exp(-beta_times_theta_x_i))
    likelihood_function_value = likelihood_function_terms.sum()
    return likelihood_function_value


def update_parameters(data: Tuple[np.ndarray, np.ndarray], parameters_in: np.ndarray, gamma: float) -> np.ndarray:
    """
    Generate the new theta estimate, theta_n+1 = theta_n + gamma * grad(l).
    """
    gradient: np.ndarray = compute_gradient(data=data, parameters=parameters_in)
    parameters_out: np.ndarray = parameters_in + gamma * gradient
    return parameters_out


def compute_likelihood_function(xs: np.ndarray, parameters: np.ndarray):
    ys: np.ndarray = 1 / (1 + np.exp(parameters[0] * (parameters[1] - xs)))
    return ys


def plot_xy_data(data: Tuple[np.ndarray, np.ndarray], gd_parameters: np.ndarray, nr_parameters: np.ndarray):
    xs: np.ndarray = data[0]
    xs_for_sorting = xs
    xs_for_sorting.sort()
    ys: np.ndarray = data[1]

    gradient_descent_predicted_curve = compute_likelihood_function(xs=xs_for_sorting, parameters=gd_parameters)
    newton_raphson_predicted_curve = compute_likelihood_function(xs=xs_for_sorting, parameters=nr_parameters)

    plt.scatter(xs, ys, c='g', label='raw data')
    plt.plot(xs_for_sorting, gradient_descent_predicted_curve, c='r', label='gradient descent prediction')
    plt.plot(xs_for_sorting, newton_raphson_predicted_curve, c='k', label='newton raphson prediction')
    plt.show()


def descend(max_iterations: int, min_change: float, data: Tuple, initial_parameters: np.ndarray, gamma: float):
    """Descend the gradient until one of the two stop conditions is met."""
    n = 0
    change_in_estimate = 10000
    parameters = initial_parameters
    prev_likelihood_value = -10000
    curr_likelihood_val = compute_log_likelihood_function_value(data, initial_parameters)
    while n <= max_iterations and np.abs(change_in_estimate) > min_change:
        print(f"iteration: {n}")
        parameters = update_parameters(data=data, parameters_in=parameters, gamma=gamma)
        prev_likelihood_value = curr_likelihood_val
        print(f"previous likelihood value: {prev_likelihood_value}")
        curr_likelihood_val = compute_log_likelihood_function_value(data=data, parameters=parameters)
        print(f"current likelihood value: {curr_likelihood_val}")
        change_in_estimate = curr_likelihood_val - prev_likelihood_value
        n += 1
        print(f"change in estimate from iteration {n-1} to {n}: {change_in_estimate}")
        if np.abs(change_in_estimate) <= min_change:
            print(f"change converged to within sensitivity: {min_change}. Exiting.")
        elif n == max_iterations:
            print(f"At max iterations, returning.")
    return parameters


def run_gradient_descent(filename: str, max_iterations: int, min_change: float, initial_parameters: np.ndarray, gamma: float):
    data = get_xy_values(filename=filename)
    print(f"""Running gradient descend with {max_iterations} max iterations, {min_change} minimum step-to-step \\"
          change, initial parameters: {initial_parameters}, and learning rate: {gamma}""")
    final_estimate: np.ndarray = descend(
        max_iterations=max_iterations,
        min_change=min_change,
        data=data,
        initial_parameters=initial_parameters,
        gamma=gamma
    )
    return final_estimate


"""Library functions pertaining to Problem 1, part b"""


def compute_hessian(data: Tuple[np.ndarray, np.ndarray], parameters: np.ndarray) -> np.ndarray:
    """
    Compute the Hessian matrix.

    In our case, this will be a 2x2 matrix, and we will be computing the 2nd partial derivatives from their analytic
    formulas.

    """
    db2 = compute_double_beta_derivative(data=data, parameters=parameters)
    dt2 = compute_double_theta_derivative(data=data, parameters=parameters)
    mixed = compute_mixed_partial_derivative(data=data, parameters=parameters)

    hessian: np.ndarray = np.array([[db2, mixed], [mixed, dt2]])
    return hessian


def compute_double_beta_derivative(data: Tuple[np.ndarray, np.ndarray], parameters: np.ndarray) -> float:
    xs: np.ndarray = data[0]
    beta: float = parameters[0]
    theta: float = parameters[1]
    theta_minus_x_i: np.ndarray = theta - xs
    minus_beta_theta_minus_x_i: np.ndarray = -beta * theta_minus_x_i
    theta_minus_x_i_squared: np.ndarray = np.multiply(theta_minus_x_i, theta_minus_x_i)
    expo: np.ndarray = np.exp(minus_beta_theta_minus_x_i)
    numerators: np.ndarray = np.multiply(-theta_minus_x_i_squared, expo)
    denominators: np.ndarray = np.multiply(1 + expo, 1 + expo)
    double_beta_derivative_terms: np.ndarray = np.divide(numerators, denominators)
    double_beta_derivative = double_beta_derivative_terms.sum()
    return double_beta_derivative


def compute_double_theta_derivative(data: Tuple[np.ndarray, np.ndarray], parameters: np.ndarray) -> float:
    xs: np.ndarray = data[0]
    beta: float = parameters[0]
    theta: float = parameters[1]
    theta_minus_x_i: np.ndarray = theta - xs
    minus_beta_theta_minus_x_i: np.ndarray = -beta * theta_minus_x_i
    expo: np.ndarray = np.exp(minus_beta_theta_minus_x_i)
    numerators: np.ndarray = expo * -beta * beta
    denominators: np.ndarray = np.multiply(1 + expo, 1 + expo)
    double_theta_derivative_terms: np.ndarray = np.divide(numerators, denominators)
    double_theta_derivative = double_theta_derivative_terms.sum()
    return double_theta_derivative


def compute_mixed_partial_derivative(data: Tuple[np.ndarray, np.ndarray], parameters: np.ndarray) -> float:
    xs: np.ndarray = data[0]
    beta: float = parameters[0]
    theta: float = parameters[1]
    theta_minus_x_i: np.ndarray = theta - xs
    minus_beta_theta_minus_x_i: np.ndarray = -beta * theta_minus_x_i
    minus_two_beta_theta_minus_x_i: np.ndarray = 2 * minus_beta_theta_minus_x_i
    expo: np.ndarray = np.exp(minus_beta_theta_minus_x_i)
    expo_2: np.ndarray = np.exp(minus_two_beta_theta_minus_x_i)
    denominators: np.ndarray = np.multiply(1 + expo, 1 + expo)
    numerators: np.ndarray = np.multiply(1 + minus_beta_theta_minus_x_i, expo) + expo_2
    mixed_derivative_terms: np.ndarray = np.divide(numerators, denominators)
    mixed_partial_derivative = mixed_derivative_terms.sum()
    return mixed_partial_derivative


def take_newton_raphson_step(input_parameters: np.ndarray, hessian: np.ndarray, gradient: np.ndarray):
    inverse_hessian = np.linalg.inv(hessian)
    updated_parameters: np.ndarray = input_parameters - inverse_hessian.dot(gradient)
    return updated_parameters


def run_newton_raphson(filename: str, max_iterations: int, min_change: float, initial_parameters: np.ndarray):
    print(f"""Running Newton Raphson estimation on the data in file: {filename} with parameters: \\
    max_iterations = {max_iterations}, min_change = {min_change}, initial parameters = {initial_parameters}""")
    data: Tuple[np.ndarray, np.ndarray] = get_xy_values(filename=filename)
    parameters = initial_parameters
    n = 0
    change_in_estimate = 10000
    prev_likelihood_value = 10000
    curr_likelihood_value = compute_log_likelihood_function_value(data=data, parameters=parameters)
    while n <= max_iterations and np.abs(change_in_estimate) > min_change:
        print(f"N-R iteration: {n}")
        hessian = compute_hessian(data=data, parameters=parameters)
        print(f"NR - hessian = {hessian}")
        gradient = compute_gradient(data=data, parameters=parameters)
        print(f"NR - gradient = {gradient}")
        parameters = take_newton_raphson_step(input_parameters=parameters, hessian=hessian, gradient=gradient)
        prev_likelihood_value = curr_likelihood_value
        print(f"NR - previous estimate = {prev_likelihood_value}")
        curr_likelihood_value = compute_log_likelihood_function_value(data=data, parameters=parameters)
        print(f"NR - current estimate = {curr_likelihood_value}")
        change_in_estimate = curr_likelihood_value - prev_likelihood_value
        print(f"NR - change in estimate from last step: {change_in_estimate}")
        n += 1
        if np.abs(change_in_estimate) <= min_change:
            print(f"change converged to within sensitivity: {min_change}. Exiting.")
        elif n == max_iterations:
            print(f"At max iterations, returning.")

    return parameters


"""Library functions pertaining to Problem 2 start here."""


class CountriesData:
    """Class to hold data and methods related to infection rate statistics for different countries"""
    class CountryData:
        """Class to hold data and methods for infection rate statistics for a specific country."""
        def __init__(self, day_numbers: np.ndarray, dates: np.ndarray, new_infection_data: np.ndarray):
            """Class constructor for CountryData"""
            self.day_numbers = day_numbers
            self.dates = dates
            self.new_infection_data = new_infection_data
            self.infection_rate_estimate: np.ndarray = np.array([])

        def compute_estimates(self):
            """Estimate the infection rate using the summation formula in problem 2 part a"""
            rate_estimate_list = []
            for tau in range(len(self.new_infection_data) - 11):
                estimate = 0
                for j in range(7):
                    estimate += self.new_infection_data[tau + j + 4] / self.new_infection_data[tau + j]
                rate_estimate_list.append(estimate)
            rate_estimates = np.array(rate_estimate_list)
            self.infection_rate_estimate = rate_estimates

    def __init__(self, filename):
        """Class constructor for CountriesData"""
        self.data = dict()
        self.get_infection_data(filename=filename)

    def get_infection_data(self, filename):
        """Process the infection rate data from the csv file."""
        with open(filename, "r") as f:
            lines = f.readlines()

            line: str = lines[0]
            dates = line.split(",")
            date_list = [d for d in dates[1:]]
            day_number_list = [j for j, d in enumerate(dates[1:])]
            date_data: np.ndarray = np.array(date_list)
            day_number_data: np.ndarray = np.array(day_number_list)

            for i, line in enumerate(lines[1:]):
                l = line.strip(",\n")
                country = l.split(",")[0]
                data_as_list = l.split(",")[1:]
                data = np.array([int(s) for s in data_as_list])
                country_data = self.CountryData(day_numbers=day_number_data, dates=date_data, new_infection_data=data)
                self.data[country] = country_data

    def get_estimates(self):
        """Estimate the infection rates for each CountryData object in the data class dictionary."""
        for k, v in self.data.items():
            v.compute_estimates()

    def get_estimates_for_country(self, country):
        """Compute estimates for a specific country by name."""
        self.data[country].compute_estimates()

    def perform_test_for_country(self, country, tau_x, tau_y):
        """Execute the sign test exactly as described in problem 2, parts a and b."""
        country_data = self.data[country]
        signs = []
        for j in range(7):
            diff = country_data.infection_rate_estimate[tau_x + j] - country_data.infection_rate_estimate[tau_y + j]
            if diff >= 0:
                signs.append(1)
            else:
                signs.append(-1)
        signs = np.array(signs) + 1
        test_statistic = int(round((1/2) * signs.sum()))
        print(f"test_statistic for {country} = {test_statistic}")
        probability_of_outcome_under_null_hyp = \
            self.compute_probability_of_test_statistic_or_more_extreme_result_under_null_hypothesis(
                test_statistic=test_statistic
            )
        print(f"probability of outcome = {probability_of_outcome_under_null_hyp}")
        return probability_of_outcome_under_null_hyp

    def compute_probability_of_test_statistic_or_more_extreme_result_under_null_hypothesis(self, test_statistic):
        """Compute the probability of the computed test statistic or a more extreme value."""
        comb_sum = 0
        for k in range(test_statistic, 7 + 1):
            comb_sum += comb(7, k=k, exact=True)
        probability_of_test_statisitc_or_more_extreme = comb_sum * np.power((1/2), 7)
        return probability_of_test_statisitc_or_more_extreme
