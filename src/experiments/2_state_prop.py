from numpy import array, linspace
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

from src.agents.util import GaussianRegression, GaussianRegression2, BayesTestRegression

plt.rcParams.update({'font.size': 24})

scale=1
state_2_posterior = norm(loc=1, scale=scale)

models_mean = [BayesTestRegression(dim=1), GaussianRegression2(dim=1)]
models_sample = [BayesTestRegression(dim=1), GaussianRegression2(dim=1)]

for i in range(1000):
    target = state_2_posterior.rvs()
    for model in models_sample:
        model.update_posterior(array([1]), target, n=1)
    for model in models_mean:
        model.update_posterior(array([1]), array([1]), n=1)

x = linspace(1-3*scale, 1+3*scale, 5000)

def plot_posterior(model, ax, title, known_noise=True):
    ax.set_title(title)
    ax.set_xticks(np.linspace(1-3*scale, 1+3*scale, 5)[1:-1])
    ax.set_xlim(1-3*scale, 1+3*scale)
    ax.set_ylim(0, norm.pdf(0, 0, scale)*1.1)

    param_samples = [model.sample_params() for _ in range(len(x))]

    if known_noise:
        samples = np.array([p for p in param_samples])# + np.random.normal(0, np.sqrt(model.noise)) for p in param_samples])

    else:
        samples = np.array([p[0] + np.random.normal(0, np.sqrt(p[1])) for p in param_samples])
    sns.kdeplot(samples.reshape(-1), label="Posterior Samples", legend=False, ax=ax)

    # ax.plot(x, [model.pdf(i, np.array([1])) for i in x], linewidth=2, label="State 1 Posterior PDF")
    ax.plot(x, state_2_posterior.pdf(x), linewidth=2, label="State 2 posterior")


fig, axs = plt.subplots(1,4)
plt.subplots_adjust(bottom=0.15, wspace=0.4)


plot_posterior(models_mean[0], axs[0], title="Known Noise\nMean Target")
plot_posterior(models_sample[0], axs[1], title="Known Noise\nSample Target")
plot_posterior(models_mean[1], axs[2], title="Unknown Noise\nMean Target", known_noise=False)
plot_posterior(models_sample[1], axs[3], title="Unknown Noise\nSample Target", known_noise=False)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=False)

fig.add_subplot(111, frameon=False)

# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.xlabel('State Value')
plt.ylabel('Probability', labelpad=80)
plt.show()

models_sample[1].print_parameters()

plt.show()
