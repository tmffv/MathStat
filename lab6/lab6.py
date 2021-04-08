import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt



def MNK(x, y):
    beta_ls = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    alpha_ls = np.mean(y) - beta_ls * np.mean(x)
    return alpha_ls, beta_ls


def MNM(x, y, initial_guess):
    functionToMinimize = lambda beta: np.sum(np.abs(y - beta[0] - beta[1] * x))
    result = opt.minimize(functionToMinimize, initial_guess)
    alpha_lm = result['x'][0]
    beta_lm = result['x'][1]
    return alpha_lm, beta_lm


def coefficientEstimates(x, y):
    alpha_ls, beta_ls = MNK(x, y)
    alpha_lm, beta_lm = MNM(x, y, np.array([alpha_ls, beta_ls]))
    return alpha_ls, beta_ls, alpha_lm, beta_lm


def graphRegression(x, y, type, estimates):
    alpha_ls, beta_ls, alpha_lm, beta_lm = estimates
    plt.plot(x, x * (2 * np.ones(len(x))) + 2 * np.ones(len(x)), label='Модель', color='red')
    plt.plot(x, x * (beta_ls * np.ones(len(x))) + alpha_ls * np.ones(len(x)), label='МНК', color='black')
    plt.plot(x, x * (beta_lm * np.ones(len(x))) + alpha_lm * np.ones(len(x)), label='МНМ', color='blue')
    plt.scatter(x, y, label="Выборка", facecolors='none', edgecolors='black')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-1.8, 2])
    plt.legend()
    plt.savefig(type + '.png', format='png')
    plt.close()


def criteriaComparison(x, estimates):
    alpha_ls, beta_ls, alpha_lm, beta_lm = estimates
    model = lambda x: 2 + 2 * x
    lsc = lambda x: alpha_ls + beta_ls * x
    lmc = lambda x: alpha_lm + beta_lm * x
    sum_ls, sum_lm = 0, 0
    for point in x:
        y_ls = lsc(point)
        y_lm = lmc(point)
        y_model = model(point)
        sum_ls += pow(y_model - y_ls, 2)
        sum_lm += pow(y_model - y_lm, 2)
    print('sum_ls =', sum_ls, " < ", 'sum_lm =', sum_lm) if sum_ls < sum_lm \
        else print('sum_lm =', sum_lm, " < ", 'sum_ls =', sum_ls)


if __name__ == "__main__":
    x = np.linspace(-1.8, 2, 20)
    y = 2 + 2 * x + stats.norm(0, 1).rvs(20)
    for type in ['Without perturbations', 'With perturbations']:
        estimates = coefficientEstimates(x, y)
        alpha_ls, beta_ls, alpha_lm, beta_lm = estimates
        print(type)
        print("МНК:")
        print('alpha_ls = ' + str(np.around(alpha_ls, decimals=2)))
        print('beta_ls = ' + str(np.around(beta_ls, decimals=2)))
        print("МНМ:")
        print('alpha_lm = ' + str(np.around(alpha_lm, decimals=2)))
        print('beta_lm = ' + str(np.around(beta_lm, decimals=2)))
        graphRegression(x, y, type, estimates)
        criteriaComparison(x, estimates)
        y[0] += 10
        y[-1] -= 10