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
