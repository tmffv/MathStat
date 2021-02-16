from scipy.stats import norminvgauss, laplace, poisson, cauchy, uniform
import numpy as np
import matplotlib.pyplot as plt
import math

sizes = [10, 50, 1000]


def normalNumbers():
    for size in sizes:
        fig, ax = plt.subplots(1, 1)
        ax.hist(norminvgauss.rvs(1, 0, size=size), histtype='stepfilled', alpha=0.5, color='blue', density=True)
        x = np.linspace(norminvgauss(1, 0).ppf(0.01), norminvgauss(1, 0).ppf(0.99), 100)
        ax.plot(x, norminvgauss(1, 0).pdf(x), '-')
        ax.set_title('NormalNumbers n = ' + str(size))
        ax.set_xlabel('NormalNumbers')
        ax.set_ylabel('density')
        plt.grid()
        plt.show()
    return


def cauchyNumbers():
    for size in sizes:
        fig, ax = plt.subplots(1, 1)
        ax.hist(cauchy.rvs(size=size), histtype='stepfilled', alpha=0.5, color='blue', density=True)
        x = np.linspace(cauchy().ppf(0.01), cauchy().ppf(0.99), 100)
        ax.plot(x, cauchy().pdf(x), '-')
        ax.set_title('CauchyNumbers n = ' + str(size))
        ax.set_xlabel('CauchyNumbers')
        ax.set_ylabel('density')
        plt.grid()
        plt.show()
    return


def laplaceNumbers():
    for size in sizes:
        fig, ax = plt.subplots(1, 1)
        param = 1 / math.sqrt(2)
        ax.hist(laplace.rvs(size=size, scale=1 / math.sqrt(2), loc=0), histtype='stepfilled', alpha=0.5, color='blue',
                density=True)
        x = np.linspace(laplace(scale=param, loc=0).ppf(0.01), laplace(scale=param, loc=0).ppf(0.99), 100)
        ax.plot(x, laplace(scale=param, loc=0).pdf(x), '-')
        ax.set_title('LaplaceNumbers n = ' + str(size))
        ax.set_xlabel('LaplaceNumbers')
        ax.set_ylabel('density')
        plt.grid()
        plt.show()
    return


def poisNumbers():
    for size in sizes:
        fig, ax = plt.subplots(1, 1)
        ax.hist(poisson.rvs(10, size=size), histtype='stepfilled', alpha=0.5, color='blue', density=True)
        x = np.arange(poisson.ppf(0.01, 10), poisson.ppf(0.99, 10))
        ax.plot(x, poisson(10).pmf(x), '-')
        ax.set_title('PoisNumbers n = ' + str(size))
        ax.set_xlabel('PoisNumbers')
        ax.set_ylabel('density')
        plt.grid()
        plt.show()
    return


def unifNumbers():
    for size in sizes:
        fig, ax = plt.subplots(1, 1)
        ax.hist(uniform.rvs(size=size, loc=-math.sqrt(3), scale=2 * math.sqrt(3)),
                histtype='stepfilled', alpha=0.3, color='blue', density=True)
        x = np.linspace(uniform(loc=-math.sqrt(3), scale=2 * math.sqrt(3)).ppf(0.01),
                        uniform(loc=-math.sqrt(3), scale=2 * math.sqrt(3)).ppf(0.99), 100)
        ax.plot(x, uniform(loc=-math.sqrt(3), scale=2 * math.sqrt(3)).pdf(x), '-')
        ax.set_title('UnifNumbers n = ' + str(size))
        ax.set_xlabel('UnifNumbers')
        ax.set_ylabel('density')
        plt.grid()
        plt.show()
    return


normalNumbers()
cauchyNumbers()
laplaceNumbers()
poisNumbers()
unifNumbers()
