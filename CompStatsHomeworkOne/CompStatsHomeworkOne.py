"""Computational Statistics and Data Analysis Homework 1 Python Code
    Author: Ryan Hutchins
"""
import numpy as np
import math
import matplotlib.pyplot as plt

from typing import List, Tuple
from enum import Enum


class Distribution(Enum):
    """An enum of named distributions from which we can draw samples."""
    Normal = "normal"
    Poisson = "poisson"


def sample_from_distribution(
        distribution: Distribution,
        sample_size: int,
        parameters: Tuple
) -> np.ndarray:
    """ Sample from a named distribution.

    :param distribution: The type of distribution from which to sample
    :param sample_size: The number of samples to draw from the distribution
    :param parameters: the parameters of the distribution
    :return: 1-dimensiond numpy ndarray of length sample_size
    """
    if len(parameters) == 1:
        return function_selector[distribution](parameters[0], sample_size)
    elif len(parameters) == 2:
        return function_selector[distribution](parameters[0], parameters[1], sample_size)
    else:
        print("Currently, no 3-parameter distributions are supported.")


def compute_sample_mean(data: np.ndarray) -> float:
    """ Takes a 1-D numpy ndarray and computes the mean of its values.

    :param data: a 1-D numpy ndarray
    :return: The mean of the values in data
    """
    return data.mean()


def compute_sample_variance(data: np.ndarray) -> float:
    """ Takes a 1-D numpy ndarray and computes the variance of its values.

    :param data: a 1-D numpy ndarray
    :return: The mean of the values in data
    """
    return data.var()


def compute_sample_standard_deviation(data: np.ndarray) -> float:
    """ Takes a 1-D numpy ndarray and computes the standard deviation of its values.

        :param data: a 1-D numpy ndarray
        :return: The mean of the values in data
        """
    return math.sqrt(compute_sample_variance(data))


def compute_sample_variance_using_formula_from_part_b_1(data: np.ndarray) -> float:
    """Computes the sample variance as: (1/N) SUM (x_i - mu)^2"""
    N = len(data)
    sample_mean: float = data.mean()
    data_as_mean_offset_list: List = [it for it in map(lambda x: (x - sample_mean)*(x - sample_mean), data.tolist())]
    data_as_mean_offset_ndarray = np.array(data_as_mean_offset_list)
    sum_of_mean_offsets = data_as_mean_offset_ndarray.sum()
    sample_variance = (1/N)*sum_of_mean_offsets
    return sample_variance


def compute_sample_variance_using_formula_from_part_b_2(data: np.ndarray) -> float:
    """Computes the sample variance as: [(1/n) SUM (x_i)^2] - mu^2"""
    N = len(data)
    sample_mean: float = data.mean()
    data_squared = [it for it in map(lambda x: x*x, data.tolist())]
    data_squared_np = np.array(data_squared)
    average_data_squared = (1/N)*data_squared_np.sum()
    sample_variance = average_data_squared - (sample_mean * sample_mean)
    return sample_variance


def compute_difference_between_true_param_and_estimate(true_param: float, estimate: float) -> float:
    """Convenience function that takes absolute value of difference between two values."""
    return math.fabs(true_param - estimate)


"""Implements the Command design pattern to call the numpy function for the desired distribution"""
function_selector = {
    Distribution.Normal: np.random.normal,
    Distribution.Poisson: np.random.poisson
}

"""Our Code library ends here. What follows is a script to apply our code library to the parts of Problem 4."""

"""Here is the code for Problem 4, Part a:"""
s = sample_from_distribution(distribution=Distribution.Normal, sample_size=1000, parameters=(2, 3))


"""Here is the code for Problem 4, Part b. 
 Note that there is a built-in np.ndarray.var() function that computes the sample variance that we could use instead of 
 implementing our own. We use this function in lieu of our own function in part c.
"""
true_variance = 9
sample_variance_method_1 = compute_sample_variance_using_formula_from_part_b_1(s)
diff_1 = compute_difference_between_true_param_and_estimate(true_variance, sample_variance_method_1)

sample_variance_method_2 = compute_sample_variance_using_formula_from_part_b_2(s)
diff_2 = compute_difference_between_true_param_and_estimate(true_variance, sample_variance_method_2)

answer = f"""PROBLEM 4, PART B: The value of the variance according to estimate method 1 is: {sample_variance_method_1}\
. The difference between the true variance and the estimate by method 1 is: {diff_1}. \
The variance using method 2 is: {sample_variance_method_2} between the true variance and method 2 is: {diff_2}"""
print(answer)
print("\n")


"""Here is the code for Problem 4, part c:"""
# First, we generate a sample S, where S_i ~ N(2, 3)
S = sample_from_distribution(Distribution.Normal, 1000, (2, 3))
# Now, we let Y_i = aS_i + b, where a = 3 and b = 5.
a = 3
b = 5
# Transform S inot U using a lambda function to implement the location-scale transform and make a list out of it using
# a list comprehension.
U = [it for it in map(lambda s: a*s + b, S.tolist())]
sample_variance_of_U = np.array(U).var()
# And we compute a^2 * Var(S):
a_squared_times_variance_of_S = a * a * S.var()
answer_part_c = f"""PROBLEM 4, PART C: The sample variance by direct computation on data is: {sample_variance_of_U} \
and the sample variance by computing a^2 * Var(S) is: {a_squared_times_variance_of_S}. The difference between the two \
is: \
{round(compute_difference_between_true_param_and_estimate(sample_variance_of_U, a_squared_times_variance_of_S), 6)} \
to six decimal places."""
print(answer_part_c)


"""Here is the code for Problem 4, part d:"""
# Generate 1000 samples from the Poisson distribution with parameter 1 through 10, returns a list of numpy arrays
s = [sample_from_distribution(Distribution.Poisson, 1000, (i,)) for i in range(1, 11)]
# Flatten the list of arrays we just generated into a single dataset:
s_concatenated = [item for sublist in s for item in sublist.tolist()]
bins = [i for i in range(1, 21)]
_ = plt.hist(s_concatenated, bins=bins, range=(0, 20))
plt.title("Problem 4, Part d")
plt.show()

