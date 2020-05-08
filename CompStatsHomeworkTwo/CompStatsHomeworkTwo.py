""" Computational Statistics and Data Analysis Homework 2 Module
    Author: Ryan Hutchins
    Institution: Ruprecht Karls Universitaet Heidelberg

    "Of all the things I've lost, it's my mind I miss the most."

"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import comb
from typing import List


def do_problem_four_part_a(sample_size: int, num_students_observed: int, p: float):
    """
        Homework 2, Problem 2, Part A

        We execute a solution to this problem in two parts. First, we compute the theoretical solution. That is to say, we
        compute an exact value for:

            P(Y >= 12 ; p = 0.7) = sum from k=12 to k=20 (20 choose k) p^(k)(1-p)^(20-k)


    """
    probability = 0
    print(probability)
    pmf_y = [comb(sample_size, k) * np.power(p, k) * np.power(1-p, sample_size-k) for k in range(0, sample_size+1)]
    cdf_y = []
    cum_prob = 0
    for i in range(0, 21):
        cum_prob += pmf_y[i]
        cdf_y.append(cum_prob)

    probability = sum(pmf_y[num_students_observed:])

    print(f"""The probability of observing at least {observation_count} students applying 
    probability is: {probability}""")

    width = 0.35
    labels = [f'X={x}' for x in range(0, 21)]
    x_pts = [x for x in range(0, 21)]

    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    rects1 = ax.bar(x - width / 2, pmf_y, width, label='PMF')
    rects2 = ax.bar(x + width / 2, cdf_y, width, label='CDF')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Probability')
    ax.set_title('PDF and CDF for Bin(n, k)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(),3)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()

    return probability


n = 20
observation_count = 12
prob = 0.7

do_problem_four_part_a(n, observation_count, prob)


class Bin:

    def __init__(self, min_point, max_point, mid_point):
        self.min_point = min_point
        self.max_point = max_point
        self.mid_point = mid_point
        self.data = []

    def get_data_point_count(self):
        return len(self.data)


def create_bins(num_bins: int, range_min: float, range_max: float):
    bin_width = (range_max - range_min)/num_bins
    bins = []
    bin_min = range_min
    for i in range(0, num_bins):
        bin_max = bin_min + bin_width
        bin_midpoint = bin_min + bin_width/2.0
        bins.append(Bin(bin_min, bin_max, bin_midpoint))
        bin_min = bin_max
    return bins


def bin_data_points(bins: List[Bin], data_points: List):
    list.sort(data_points)
    bin_index = -1
    while data_points:
        point = data_points.pop()
        for i, bn in enumerate(bins):
            if bn.min_point <= point < bn.max_point:
                bin_index = i
        bins[bin_index].data.append(point)
    return bins


def do_problem_four_part_b():
    """Take 10,000 samples from each distribution"""
    x = np.random.normal(loc=1, scale=1, size=10000)
    y = np.random.normal(loc=5, scale=2, size=10000)
    z = np.add(x, y)

    # Compute the sample means of each distribution and the sample variance of Z
    x_sample_mean = np.mean(x)
    y_sample_mean = np.mean(y)
    sample_mean = np.mean(z)
    sample_variance = np.var(z)

    x = x.tolist()
    y = y.tolist()
    z = z.tolist()

    # Bin the results into 100 equal-width bins
    bsx = create_bins(num_bins=100, range_min=-5, range_max=15)
    bsy = create_bins(num_bins=100, range_min=-5, range_max=15)
    bsz = create_bins(num_bins=100, range_min=-5, range_max=15)
    new_bsx = bin_data_points(bins=bsx, data_points=x)
    new_bsy = bin_data_points(bins=bsy, data_points=y)
    new_bsz = bin_data_points(bins=bsz, data_points=z)

    # Get normalized values for the three distributions and the midpoints of the bins on the X axis
    x_values = [b.mid_point for b in new_bsx]
    y_values_1 = [(1/10000)*b.get_data_point_count() for b in new_bsx]
    y_values_2 = [(1/10000)*b.get_data_point_count() for b in new_bsy]
    y_values_3 = [(1/10000)*b.get_data_point_count() for b in new_bsz]

    # Compute the max count for each distribution so we can draw a dotted vertical line to it
    x_max_count = max(y_values_1)
    y_max_count = max(y_values_2)
    z_max_count = max(y_values_3)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x_values, y_values_1, alpha=0.8, c="red", s=10, label=f"X ~ N(1, 1)=")
    ax1.scatter(x_values, y_values_2, alpha=0.8, c="green", s=10, label=f"Y ~ N(5, 4)")
    ax1.scatter(x_values, y_values_3, alpha=0.8, c="blue", s=10, label=f"Z = X+Y, E[Z] = {sample_mean}, Var(Z) = {sample_variance}")
    ax1.plot([x_sample_mean, x_sample_mean], [0.0, x_max_count], "r--", label="Z mean")
    ax1.plot([y_sample_mean, y_sample_mean], [0.0, y_max_count], "g--", label="Z mean")
    ax1.plot([sample_mean, sample_mean], [0.0, z_max_count], "b--", label="Z mean")

    plt.title("X~N(1,1), Y~N(5, 4), Z = X + Y")
    plt.legend(loc=2)
    plt.show()


do_problem_four_part_b()