from absl import flags, app

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import seaborn as sns

from absl import flags, app
from scipy.stats import norm
from src.agents.util import GaussianRegression2

flags.DEFINE_integer("states", 5, "Number of states")
flags.DEFINE_float("scale", 1, "SD of target")


FLAGS = flags.FLAGS

plt.rcParams.update({'font.size': 32})

def n_state_prop(models, target_scale):

    final_state_posterior = norm(loc=1, scale=target_scale)

    T=10000
    n=1
    step = 1
    for i in range(int(T/n)):

        for m, model in enumerate(models):
            if m+step < len(models):
                target = np.array([models[m+step].sample(np.array([1]), norm.rvs(size=1)) for _ in range(n)])
                model.update_posterior(np.array([1]*n), target, n=n) 

            else:
                models[m].update_posterior(np.array([1]*n), final_state_posterior.rvs(n), n=n)


    ## Plotting code

    def plot_posterior(ax, model, index):

        x = np.linspace(1-3*target_scale, 1+3*target_scale, 10000)

        ## Plot sampled posterior distribution
        # samples = np.array([model.sample(np.array([1])) for _ in range(len(x))])
        # sns.kdeplot(samples.reshape(-1), label="Posterior Samples", legend=False, ax=ax)
        ax.set_title("State" + str(index))
        ax.set_xticks(np.linspace(1-3*target_scale, 1+3*target_scale, 3)[1:-1])
        ax.set_xlim(1-3*target_scale, 1+3*target_scale)

        ax.plot(x, [model.pdf(i, np.array([1])) for i in x], linewidth=2, label="Posterior PDF")
        ax.plot(x, final_state_posterior.pdf(x), linewidth=2, label="Target")


    number_of_subplots=len(models)
    fig, axs = plt.subplots(1, number_of_subplots, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.01)

    for i, model in enumerate(models):
        plot_posterior(axs[i], model, i+1)

    plt.legend(loc='best', frameon=False)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel('State Value')
    plt.ylabel('Probability', labelpad=80)
    plt.show()
    
    # for i, model in enumerate(models):
    #     print("State", i+1)
    #     model.print_parameters()

def main(argv):

    target_scale = FLAGS.scale
    
    models = []
    [models.append(GaussianRegression2(dim=1)) for _ in range(FLAGS.states)]
    n_state_prop(models, target_scale)

if __name__ == '__main__':

    app.run(main)
