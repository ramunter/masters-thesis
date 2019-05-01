from numpy import array, linspace, max
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

from src.agents.util import GaussianRegression, GaussianRegression2

scale = 1
state_2_posterior = norm(loc=1, scale=scale)

model1 = GaussianRegression2(dim=1)
model2 = GaussianRegression2(dim=1)

T=10000
n=1

for i in range(int(T/n)):

    # var1 = array([1]*n)@model1.cov@array([1]*n).T +\
    #         model1.b/(model1.a - 1)
    # var2 = array([1]*n)@model2.cov@array([1]*n).T +\
    #         model2.b/(model2.a - 1)

    target = array([model2.sample(array([1])) for _ in range(n)])

    # if var1 > var2:
    # if i > 10:
    model1.update_posterior(array([1]*n),target, n=n) 

    model2.update_posterior(array([1]*n), state_2_posterior.rvs(n), n=n)

x = linspace(1-3*scale, 1+3*scale, 10000)

def plot_posterior(model, plot_index, title):
    plt.subplot(plot_index)
    samples = array([model.sample(array([1])) for _ in range(len(x))])
    sns.kdeplot(samples.reshape(-1), label="Posterior Samples", legend=False)
    
    plt.plot(x, [model.pdf(i, np.array([1])) for i in x], label="Posterior PDF")
    plt.plot(x, state_2_posterior.pdf(x), label="Target")

    plt.title(title)
    plt.xlim(1-3*scale, 1+3*scale)
    plt.xlabel('State Value')
    plt.ylabel('Probability')

# plot_posterior(models_mean[0], 221, title="Known Noise - Mean Target")
plot_posterior(model1, 121, title="State1")
plot_posterior(model2, 122, title="State2")
plt.legend(loc='best', frameon=False)
plt.show()

model1.print_parameters()
model2.print_parameters()


