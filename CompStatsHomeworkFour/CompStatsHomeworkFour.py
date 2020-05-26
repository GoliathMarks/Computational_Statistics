"""
    Computational Statistics Homework 4
    Author: Ryan Hutchins
    University of Heidelberg, Summer Somester 2020
"""
import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple


def get_xy_values(filename: str) -> Tuple[np.ndarray, np.ndarray]:
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
    print(f"beta = {parameters[0]}, theta = {parameters[1]}")
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
    print(f"gradient = {gradient}")
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


def plot_xy_data(data: Tuple[np.ndarray, np.ndarray], parameters: np.ndarray):
    xs: np.ndarray = data[0]
    xs_for_sorting = xs
    print(f"xs = {xs}")
    xs_for_sorting.sort()
    print(f"xs_sorted = {xs}")
    ys: np.ndarray = data[1]

    predicted_curve = compute_likelihood_function(xs=xs_for_sorting, parameters=parameters)

    print(predicted_curve)

    plt.scatter(xs, ys, c='g', label='raw data')
    plt.plot(xs_for_sorting, predicted_curve, c='r', label='predicted logistic curve')
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


def run(filename: str, max_iterations: int, min_change: float, initial_parameters: np.ndarray, gamma: float):
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


fn = "/home/ryan/PycharmProjects/ComputationalStatistics/CompStatsHomeworkFour/data/ex4task1.csv"
d = get_xy_values(filename=fn)
#print(d)
beta_ = 1
theta_ = 1
params = np.array([beta_, theta_])

final = run(filename=fn, max_iterations=500, min_change=0.01, initial_parameters=params, gamma=0.0001)

#print(compute_gradient(data=d, parameters=params))
#print(compute_likelihood_function_value(data=d, parameters=params))
print(f"final parameter estimates: beta = {final[0]}, theta = {final[1]}")
plot_xy_data(data=d, parameters=final)
