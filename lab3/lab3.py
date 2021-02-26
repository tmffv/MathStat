import numpy as np
import math
import matplotlib.pyplot as plt
from tabulate import tabulate

distributions = {
     'Normal': lambda num: np.random.normal(0, 1, num),
     'Cauchy': lambda num: np.random.standard_cauchy(num),
     'Laplace': lambda num: np.random.laplace(0, math.sqrt(2) / 2, num),
     'Pois': lambda num: np.random.poisson(10, num),
     'Unif': lambda num: np.random.uniform(-math.sqrt(3), math.sqrt(3), num)
}


def get_distribution(distr_name, num):
    return distributions.get(distr_name)(num)


def theoretical_prob(sample):
    min = np.quantile(sample, 0.25) - 1.5 * (np.quantile(sample, 0.75) - np.quantile(sample, 0.25))
    max = np.quantile(sample, 0.75) + 1.5 * (np.quantile(sample, 0.75) - np.quantile(sample, 0.25))
    return min, max


def ejection_num(sample, min, max):
    ejection = 0
    for elem in sample:
        if elem < min or elem > max:
            ejection += 1
    return ejection


if __name__ == "__main__":
    for distr_name in distributions.keys():
        sample_20 = get_distribution(distr_name, 20)
        sample_100 = get_distribution(distr_name, 100)
        plt.boxplot((sample_20, sample_100), patch_artist=True, boxprops=dict(facecolor='blue'),
                    labels=[20, 100])
        plt.xlabel("n")
        plt.ylabel("x")
        plt.title(distr_name, fontweight="bold")
        plt.show()

    rows = []

    for distr_name in distributions.keys():
        ejection_20 = 0
        ejection_100 = 0
        for i in range(1000):
            sample_20 = get_distribution(distr_name, 20)
            sample_100 = get_distribution(distr_name, 100)
            min_20, max_20 = theoretical_prob(sample_20)
            min_100, max_100 = theoretical_prob(sample_100)
            ejection_20 += ejection_num(sample_20, min_20, max_20)
            ejection_100 += ejection_num(sample_100, min_100, max_100)

        rows.append([distr_name + ", n = 20", np.around(ejection_20 / 1000 / 20, decimals=2)])
        rows.append([distr_name + ", n = 100", np.around(ejection_100 / 1000 / 100, decimals=2)])

    print(tabulate(rows))
    print("\n")