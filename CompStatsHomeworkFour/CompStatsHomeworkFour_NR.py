"""
This file is just used to run code that is sitting in the module CompStatsHomeworkFour, in the file:
    CompStatsHomeworkFourLib.py

CSDA Homework 4
University of Heidelberg
Summer "Semester" 2020 -- wherein we all learned to balk at human contact and embrace distance learning.
Author: Ryan Hutchins
Module for Problem 1 part b, wherein we implement components for the Newton-Raphson Method.

Fezzik: "Why do you wear a mask? Were you burned with acid, or something like that?"
Westley: Oh no. It’s just they’re terribly comfortable. I think everyone will be wearing them in the future."

-The Princess Bride

"""
import CompStatsHomeworkFour.CompStatsHomeworkFourLib as cs4
import numpy as np





fn = "/Users/administrator/PycharmProjects/ComputationalStatistics/CompStatsHomeworkFour/data/ex4task1.csv"
data = cs4.get_xy_values(filename=fn)

gd_parameters = cs4.run_gradient_descent(
    filename=fn,
    max_iterations=500,
    min_change=0.01,
    initial_parameters=np.array([1, 1]),
    gamma=0.0001
)
nr_parameters = cs4.run_newton_raphson(
    filename=fn,
    max_iterations=500,
    min_change=0.001,
    initial_parameters=np.array([1, 1])
)

print(f"gradient descent estimates: {gd_parameters}")
print(f"Newton-Raphson estimates: {nr_parameters}")

cs4.plot_xy_data(data=data, gd_parameters=gd_parameters, nr_parameters=nr_parameters)
