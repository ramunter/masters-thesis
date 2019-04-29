from numpy import array, linspace, max
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from src.agents.util import GaussianRegression, GaussianRegression2

scale=0.1
state_2_posterior = norm(loc=1, scale=scale)

model1 = GaussianRegression2(dim=1)
model2 = GaussianRegression2(dim=1)

T=1000
n=1

for i in range(int(T/n)):
    model1.posterior = norm(loc=model1.mean,
                        scale=(model1.b/(model1.a-1) + 
                                1/model1.invcov*
                                model1.b/(model1.a-1)))

    model2.posterior = norm(loc=model2.mean,
                        scale=(model2.b/(model2.a-1) + 
                                1/model2.invcov*
                                model2.b/(model2.a-1)))


    var1 = array([1]*n)@model1.cov@array([1]*n).T +\
            model1.b/(model1.a - 1)
    var2 = array([1]*n)@model2.cov@array([1]*n).T +\
            model2.b/(model2.a - 1)
    print("1", var1)
    print("2", var2)
    if abs(var2-var1)>1e-2:
        target = model2.posterior.rvs(n)
        model1.update_posterior(array([1]*n),target, n=n) #model2.posterior.rvs(n), n=n)

    # else:
    #     target = array([model1.mean[:,0]])

    model2.update_posterior(array([1]*n), state_2_posterior.rvs(n), n=n)


x = linspace(0, 2, 10000)

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


