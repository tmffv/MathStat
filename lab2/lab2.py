# coding=utf-8
import numpy as np
import scipy.stats as stats
import math
from tabulate import tabulate

sizes = [10, 100, 1000]
names = ['normal', 'cauchy', 'laplace', 'pois', 'unif']

distributions = {
     'normal': lambda num: np.random.normal(0, 1, num),
     'cauchy': lambda num: np.random.standard_cauchy(num),
     'laplace': lambda num: np.random.laplace(0, math.sqrt(2) / 2, num),
     'pois': lambda num: np.random.poisson(10, num),
     'unif': lambda num: np.random.uniform(-math.sqrt(3), math.sqrt(3), num)
}

def get_distribution(distr_name, num):
    return distributions.get(distr_name)(num)

def get_quartil(sample_sorted, p):
    return np.percentile(sample_sorted, p*100)


for distr_name in names:
    field_names = ["name", "mean", "median", "z_R", "z_Q", "z_tr"]
    rows = []
    for size in sizes:
        mean, med_x, z_R, z_Q, z_tr = [], [], [], [], []
        for i in range(1000):
            sample = get_distribution(distr_name, size)
            sample_sorted = np.sort(sample)
            mean.append(np.mean(sample))
            med_x.append(np.median(sample))
            z_R.append((sample_sorted[0] + sample_sorted[-1]) / 2)
            z_Q.append((get_quartil(sample, 1/4) + get_quartil(sample, 3/4)) / 2)
            z_tr.append(stats.trim_mean(sample, 0.25))
        rows.append([distr_name + " E(z) " + "n = " + str(size),
                          np.around(np.mean(mean), decimals=6),
                          np.around(np.mean(med_x), decimals=6),
                          np.around(np.mean(z_R), decimals=6),
                          np.around(np.mean(z_Q), decimals=6),
                          np.around(np.mean(z_tr), decimals=6)])
        rows.append([distr_name + " D(z) " + "n = " + str(size),
                          np.around(np.std(mean) * np.std(mean), decimals=6),
                          np.around(np.std(med_x) * np.std(med_x), decimals=6),
                          np.around(np.std(z_R) * np.std(z_R), decimals=6),
                          np.around(np.std(z_Q) * np.std(z_Q), decimals=6),
                          np.around(np.std(z_tr) * np.std(z_tr), decimals=6)])
    table = rows
    print(tabulate(table, field_names, tablefmt="latex"))