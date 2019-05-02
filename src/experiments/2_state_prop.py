from numpy import array, linspace
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

from src.agents.util import GaussianRegression, GaussianRegression2

scale=10
state_2_posterior = norm(loc=1, scale=scale)

models_mean = [GaussianRegression(dim=1), GaussianRegression2(dim=1)]
models_sample = [GaussianRegression(dim=1), GaussianRegression2(dim=1)]

for i in range(1000):
    target = state_2_posterior.rvs()
    for model in models_sample:
        model.update_posterior(array([1]), target, n=1)
    for model in models_mean:
        model.update_posterior(array([1]), array([1]), n=1)

x = linspace(1-3*scale, 1+3*scale, 50000)

def plot_posterior(model, plot_index, title, known_noise=True):
    plt.subplot(plot_index)
    param_samples = [model.sample() for _ in range(len(x))]

    if known_noise:
        samples = np.array([p + np.random.normal(0, np.sqrt(model.noise)) for p in param_samples])

    else:
        samples = np.array([p[0] + np.random.normal(0, np.sqrt(p[1])) for p in param_samples])

    sns.kdeplot(samples.reshape(-1), label="State 1 posterior estimate", legend=False)
    sns.lineplot(x, state_2_posterior.pdf(x), label="State 2 posterior", legend=False)
    plt.title(title)
    plt.xlim((1-3*scale, 1+3*scale))
    plt.xlabel('State Value')
    plt.ylabel('Probability')

plot_posterior(models_mean[0], 221, title="Known Noise - Mean Target")
plot_posterior(models_sample[0], 222, title="Known Noise - Sample Target")
plot_posterior(models_mean[1], 223, title="Unknown Noise - Mean Target", known_noise=False)
plot_posterior(models_sample[1], 224, title="Unknown Noise - Sample Target", known_noise=False)

plt.legend(loc='best', frameon=False)

models_sample[1].print_parameters()

plt.show()
