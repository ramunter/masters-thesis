from absl import flags, app

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from absl import flags, app
from scipy.stats import norm
from src.agents.util import GaussianRegression2

flags.DEFINE_integer("states", 5, "Number of states")
flags.DEFINE_float("scale", 1, "SD of target")
flags.DEFINE_string("plot_name", None, "Plot file name")

FLAGS = flags.FLAGS

def n_state_prop(models, target_scale):

    final_state_posterior = norm(loc=1, scale=target_scale)

    T=10000
    n=1

    for i in range(int(T/n)):

        for m, model in enumerate(models[0:-1]):
            var = np.array([1]*n)@model.cov@np.array([1]*n).T +\
                model.b/(model.a - 1) 
            vartar = np.array([1]*n)@models[m+1].cov@np.array([1]*n).T +\
                    models[m+1].b/(models[m+1].a - 1) 

            if True:#vartar < var:
                target = np.array([models[m+1].sample(np.array([1])) for _ in range(n)])
                model.update_posterior(np.array([1]*n), target, n=n) 

        models[-1].update_posterior(np.array([1]*n), final_state_posterior.rvs(n), n=n)


    def plot_posterior(model, index):

        x = np.linspace(1-3*target_scale, 1+3*target_scale, 10000)

        plt.subplot(1, len(models), index)

        samples = np.array([model.sample(np.array([1])) for _ in range(len(x))])
        sns.kdeplot(samples.reshape(-1), label="Posterior Samples", legend=False)
        plt.plot(x, [model.pdf(i, np.array([1])) for i in x], label="Posterior PDF")
        plt.plot(x, final_state_posterior.pdf(x), label="Target")

        plt.title("State" + str(index))
        plt.xlim(1-3*target_scale, 1+3*target_scale)
        plt.xlabel('State Value')
        plt.ylabel('Probability')

    plt.subplots_adjust(wspace=0.000, hspace=0.000)
    number_of_subplots=len(models)

    for i, model in enumerate(models):
        plot_posterior(model, i+1)

    plt.legend(loc='best', frameon=False)
    plt.show()

    for i, model in enumerate(models):
        print("State", i+1)
        model.print_parameters()

def main(argv):

    target_scale = FLAGS.scale
    
    models = []
    [models.append(GaussianRegression2(dim=1)) for _ in range(FLAGS.states)]
    n_state_prop(models, target_scale)

if __name__ == '__main__':

    app.run(main)
