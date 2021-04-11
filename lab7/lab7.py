import numpy as np
import scipy.stats as stats
from tabulate import tabulate


START, END = -1.5, 1.5
N_SIZE = 20
ALPHA = 0.05
_K = 5


def quantileChi2(sample, mu, sigma):
    hypothesis = lambda x: stats.norm.cdf(x, loc=mu, scale=sigma)

    borders = np.linspace(START, END, num=_K - 1)

    probabilities = np.array(hypothesis(START))
    quantities = np.array(len(sample[sample < START]))

    for i in range(_K - 2):
        p_i = hypothesis(borders[i + 1]) - hypothesis(borders[i])
        probabilities = np.append(probabilities, p_i)
        n_i = len(sample[(sample < borders[i + 1]) & (sample >= borders[i])])
        quantities = np.append(quantities, n_i)

    probabilities = np.append(probabilities, 1 - hypothesis(END))
    quantities = np.append(quantities, len(sample[sample >= END]))

    chi2 = np.divide(
        np.multiply(
            (quantities - N_SIZE * probabilities),
            (quantities - N_SIZE * probabilities)
        ),
        probabilities * N_SIZE
    )

    quantile = stats.chi2.ppf(0.95, _K - 1)
    isAccepted = True if quantile > np.sum(chi2) else False
    return chi2, isAccepted, borders, probabilities, quantities

def MLE(sample):
    mu_ml = np.mean(sample)
    sigma_ml = np.std(sample)
    print("mu_ml = ", np.around(mu_ml, decimals=2),
          " sigma_ml=", np.around(sigma_ml, decimals=2))
    return mu_ml, sigma_ml

def buildTable(chi2, borders, probabilities, quantities):
    headers = ["i", "Borders", "n_i", "p_i",
               "np_i$", "n_i - np_i", "(n_i - np_i)^2/np_i"]
    rows = []
    for i in range(0, len(quantities)):
        if i == 0:
            limits = ["-inf", np.around(borders[0], decimals=2)]
        elif i == len(quantities) - 1:
            limits = [np.around(borders[-1], decimals=2), "inf"]
        else:
            limits = [np.around(borders[i - 1], decimals=2), np.around(borders[i], decimals=2)]
        rows.append(
            [i + 1,
             limits,
             quantities[i],
             np.around(probabilities[i], decimals=4),
             np.around(probabilities[i] * N_SIZE, decimals=2),
             np.around(quantities[i] - N_SIZE * probabilities[i], decimals=2),
             np.around(chi2[i], decimals=2)]
        )
    rows.append(["SUM", "-----", np.sum(quantities), np.around(np.sum(probabilities), decimals=4),
                 np.around(np.sum(probabilities * N_SIZE), decimals=2),
                 -np.around(np.sum(quantities - N_SIZE * probabilities), decimals=2),
                 np.around(np.sum(chi2), decimals=2)]
    )
    return tabulate(rows, headers, tablefmt="latex_raw")

if __name__ == '__main__':
    normal_sample = np.random.normal(0, 1, size=N_SIZE)
    mu_ml, sigma_ml = MLE(normal_sample)
    chi2, isAccepted, borders, probabilities, quantities = quantileChi2(normal_sample, mu_ml, sigma_ml)
    print(buildTable(chi2, borders, probabilities, quantities))
    # print("Гипотеза принята") if isAccepted else print("Гипотеза не принята")
