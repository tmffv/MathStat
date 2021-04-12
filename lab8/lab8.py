import numpy as np
import scipy.stats as stats

ALPHA = 0.05
N_SIZE = [20, 100]


def muConfidenceInterval(sample):
    mu = np.mean(sample)
    sigma = np.std(sample)
    n = len(sample)
    ci = sigma * stats.t.ppf(1 - ALPHA/2, n - 1) / np.sqrt(n - 1)
    return np.around(mu - ci, decimals=2), np.around(mu + ci, decimals=2)


def sigmaConfidenceInterval(sample):
    sigma = np.std(sample)
    n = len(sample)
    ci_start = sigma * np.sqrt(n / stats.chi2.ppf(1 - ALPHA/2, n - 1))
    ci_end = sigma * np.sqrt(n/stats.chi2.ppf(ALPHA/2, n - 1))
    return np.around(ci_start, decimals=2), np.around(ci_end, decimals=2)


def muConfidenceIntervalAsymptotic(sample):
    mu = np.mean(sample)
    sigma = np.std(sample)
    n = len(sample)
    ci = sigma * stats.norm.ppf(1 - ALPHA/2) / np.sqrt(n)
    return np.around(mu - ci, decimals=2), np.around(mu + ci, decimals=2)


def sigmaConfidenceIntervalAsymptotic(sample):
    sigma = np.std(sample)
    n = len(sample)
    m4 = stats.moment(sample, 4)
    e = m4 / pow(float(sigma), 4) - 3
    U = stats.norm.ppf(1 - ALPHA/2) * np.sqrt((e + 2)/n)
    ci_start = sigma / np.sqrt(1 + U)
    ci_end = sigma / np.sqrt(1 - U)
    return np.around(ci_start, decimals=2), np.around(ci_end, decimals=2)


if __name__ == "__main__":
    for num in N_SIZE:
        sample = np.random.normal(0, 1, size=num)
        print('N = ', num)
        print('Mean', muConfidenceInterval(sample))
        print('Variance', sigmaConfidenceInterval(sample))
        print('Asymptotic_mean', muConfidenceIntervalAsymptotic(sample))
        print('Asymptotic_variance', sigmaConfidenceIntervalAsymptotic(sample))