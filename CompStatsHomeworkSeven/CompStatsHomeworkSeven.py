from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy.linalg import inv
from torch import Tensor
from torch.nn import Sequential
from typing import List
from typing import Tuple


activation_function_dictionary = {
    "linear": torch.nn.Linear,
    "relu": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid
}


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


def capture_data_as_torch_tensors(filename) -> Tuple[Tensor, Tensor]:
    # Create random Tensors to hold inputs and outputs
    with open(filename, "r") as f:
        lines: List[str] = f.readlines()[1:]
        xs: np.ndarray = np.array([float(line.split(",")[0]) for line in lines])
        ys: np.ndarray = np.array([float(line.split(",")[1].strip()) for line in lines])
        x = torch.from_numpy(xs).reshape((200, 1))
        y = torch.from_numpy(ys).reshape((200, 1))
    return x, y


def train_model(x, y, model: Sequential, loss_fn, learning_rate) -> Tuple[Sequential, float]:
    loss = None
    for t in range(20000):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x.float()).double()

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y).double()
        if t % 10000 == 9999:
            print(t, loss.item())

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors which requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    return model, loss.item()


def plot_results(x, y, y_predicted):
    plt.scatter(x.detach().numpy(), y.detach().numpy(), color="b", label="raw data")
    plt.plot(x.detach().numpy(), y_predicted.detach().numpy(), color="r", label="predicted")
    plt.legend()
    plt.show()


def construct_neural_network_model(
        input_dimension: int,
        output_dimension: int,
        hidden_layers: List[Tuple[int, str]],
) -> Sequential:
    model: Sequential = Sequential()
    print(f"input dimension = {input_dimension}")
    for i, hidden_layer in enumerate(hidden_layers):
        print(f"i = {i}, hidden layer = {hidden_layer}")
        print(f"current_layer_dimension = {hidden_layers[i][0]}")
        if i == 0:
            input_layer_module = Sequential(
                torch.nn.Linear(input_dimension, hidden_layers[i][0]),
                activation_function_dictionary[hidden_layers[i][1]]()
            )
            model.add_module("input_layer", input_layer_module)
        if i < len(hidden_layers) - 1:
            out_dim = hidden_layers[i+1][0]
            hidden_layer_module = Sequential(
                torch.nn.Linear(hidden_layers[i][0], out_dim, bias=True),
                activation_function_dictionary[hidden_layers[i][1]]()
            )
            print(f"next hidden layer dimension = {out_dim}")
        else:
            out_dim = output_dimension
            hidden_layer_module = Sequential(torch.nn.Linear(hidden_layers[i][0], out_dim, bias=True))

        model.add_module(f"hidden_layer_{i}", hidden_layer_module)

    print(model)
    return model


def execute(
        filename: str,
        input_dimension: int,
        output_dimension: int,
        hidden_layers: List[Tuple[int, str]],
        loss_fn,
        learning_rate
) -> Tuple[Sequential, float]:
    model: Sequential = construct_neural_network_model(
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        hidden_layers=hidden_layers
    )
    x, y = capture_data_as_torch_tensors(filename=filename)
    model, loss = execute_neural_network_model(x=x, y=y, model=model, loss_fn=loss_fn, learning_rate=learning_rate)
    return model, loss


def execute_neural_network_model(
        x: torch.Tensor,
        y: torch.Tensor,
        model: Sequential,
        loss_fn,
        learning_rate
) -> Tuple[Sequential, float]:
    model, loss = train_model(x=x, y=y, model=model, loss_fn=loss_fn, learning_rate=learning_rate)
    return model, loss


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


def do_part_b(filename: str):
    model, loss = execute(
        filename=filename,
        input_dimension=1,
        output_dimension=1,
        hidden_layers=[(7, "relu")],
        loss_fn=torch.nn.MSELoss(reduction='sum'),
        learning_rate=1e-5
    )

    x, y = capture_data_as_torch_tensors(filename=filename)
    y_predicted_final = model(x.float())

    plot_results(x=x, y=y, y_predicted=y_predicted_final)


def do_part_c(filename: str):
    model, loss = execute(
        filename=filename,
        input_dimension=1,
        output_dimension=1,
        hidden_layers=[(3, "relu"), (3, "relu")],
        loss_fn=torch.nn.MSELoss(reduction='sum'),
        learning_rate=1e-5
    )

    x, y = capture_data_as_torch_tensors(filename=filename)
    y_predicted_final = model(x.float())

    plot_results(x=x, y=y, y_predicted=y_predicted_final)


def do_part_d(filename: str):
    model, loss = execute(
        filename=filename,
        input_dimension=1,
        output_dimension=1,
        hidden_layers=[(7, "relu"), (7, "relu")],
        loss_fn=torch.nn.MSELoss(reduction='sum'),
        learning_rate=1e-5
    )

    x, y = capture_data_as_torch_tensors(filename=filename)
    y_predicted_final = model(x.float())

    plot_results(x=x, y=y, y_predicted=y_predicted_final)


def do_part_e(filename: str):
    model, loss = execute(
        filename=filename,
        input_dimension=1,
        output_dimension=1,
        hidden_layers=[(3, "sigmoid"), (3, "sigmoid")],
        loss_fn=torch.nn.MSELoss(reduction='sum'),
        learning_rate=1e-5
    )

    x, y = capture_data_as_torch_tensors(filename=filename)
    y_predicted_final = model(x.float())

    plot_results(x=x, y=y, y_predicted=y_predicted_final)


file = "/home/ryan/PycharmProjects/ComputationalStatistics/CompStatsHomeworkSeven/data/sheet7.csv"
#do_part_a(filename=file, degree=21)
#do_part_b(filename=file)
#do_part_c(filename=file)
do_part_d(filename=file)
#do_part_e(filename=file)
