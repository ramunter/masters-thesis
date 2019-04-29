from numpy import array, linspace, max
from scipy.stats import norm
import matplotlib.pyplot as plt

from src.agents.util import GaussianRegression, GaussianRegression2

scale=1
state_2_posterior = norm(loc=1, scale=scale)

model1 = GaussianRegression2(dim=1)
model2 = GaussianRegression2(dim=1)

for i in range(100):
    model1.posterior = norm(loc=model1.mean,
                        scale=(model1.b/(model1.a-1) + 
                                1/model1.invcov*
                                model1.b/(model1.a-1)))

    model2.posterior = norm(loc=model2.mean,
                        scale=(model2.b/(model2.a-1) + 
                                1/model2.invcov*
                                model2.b/(model2.a-1)))
    model1.update_posterior(array([1]), model2.posterior.rvs(), n=1)
    model2.update_posterior(array([1]), state_2_posterior.rvs(), n=1)


x = linspace(0.5, 1.5, 10000)

def plot_posterior(model, plot_index, title):
    plt.subplot(plot_index)
    plt.plot(x, model.posterior.pdf(x)[0,:], label="Estimate")
    plt.plot(x, state_2_posterior.pdf(x), label="Target")
    plt.title(title)
    plt.xlabel('State Value')
    plt.ylabel('Probability')

# plot_posterior(models_mean[0], 221, title="Known Noise - Mean Target")
plot_posterior(model1, 121, title="State1")
plot_posterior(model2, 122, title="State2")
plt.legend(loc='best', frameon=False)
plt.show()

model1.print_parameters()
model2.print_parameters()


