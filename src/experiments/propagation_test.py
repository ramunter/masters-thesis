from numpy import array, linspace
from scipy.stats import norm
import matplotlib.pyplot as plt

from src.agents.util import GaussianRegression, GaussianRegression2

scale=10
state_2_posterior = norm(loc=1, scale=scale)

models_mean = [GaussianRegression(dim=1), GaussianRegression2(dim=1)]
models_sample = [GaussianRegression(dim=1), GaussianRegression2(dim=1)]

for i in range(1000):
    target = state_2_posterior.rvs()
    for model in models_sample:
        model.update_posterior(array([1]), target, scale, n=1)
    for model in models_mean:
        model.update_posterior(array([1]), array([1]), scale, n=1)

models_mean[0].posterior = norm(loc=models_mean[0].mean, 
                                scale=models_mean[0].cov+models_mean[0].noise)
models_sample[0].posterior = norm(loc=models_sample[0].mean, 
                                scale=models_sample[0].cov + models_sample[0].noise)

models_mean[1].posterior = norm(loc=models_mean[1].mean,
                                scale=(models_mean[1].b/(models_mean[1].a-1) + 
                                        1/models_mean[1].invcov*
                                        models_mean[1].b/(models_mean[1].a-1)))
models_sample[1].posterior = norm(loc=models_sample[1].mean,
                                scale=(models_sample[1].b/(models_sample[1].a-1) + 
                                      1/models_sample[1].invcov*
                                      models_sample[1].b/(models_sample[1].a-1)))


x = linspace(0,2, 10000)

def plot_posterior(model, plot_index, title):
    plt.subplot(plot_index)
    plt.plot(x, model.posterior.pdf(x)[0,:], label="State 1 posterior")
    plt.plot(x, state_2_posterior.pdf(x), label="State 2 posterior")
    plt.title(title)
    plt.xlabel('State Value')
    plt.ylabel('Probability')

plot_posterior(models_mean[0], 221, title="Known Noise - Mean Target")
plot_posterior(models_sample[0], 222, title="Known Noise - Sample Target")
plot_posterior(models_mean[1], 223, title="Unknown Noise - Mean Target")
plot_posterior(models_sample[1], 224, title="Unknown Noise - Sample Target")

plt.legend(loc='best', frameon=False)

models_sample[1].print_parameters()

plt.show()
