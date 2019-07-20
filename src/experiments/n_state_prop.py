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


plt.style.use('ggplot')
plt.rcParams.update({'font.size': 32})


def n_state_prop(models, target_scale):

    final_state_posterior = norm(loc=1, scale=target_scale)

    T=5000
    step = 1
    for i in range(int(T)):
        for state in range(states):
            if state+step < states:
                X = np.array([[state,1]])
                X2 = np.array([[state+1,1]])
                var = model.var_prediction(X2)

                target = model.sample(X2, norm.rvs(size=2))
                mean_target = X2@model.mean
                td = X2@model.mean - X@model.mean
                model.update_posterior(X, target, mean_target, td, var, 1)

            else:
                target =  final_state_posterior.rvs(size=1)
                mean_target = np.array([1])
                td = 1 - X@model.mean

                model.update_posterior(X, target, mean_target, td, target_scale, 1)


    print("State", i+1)
    model.print_parameters()
    ## Plotting code

    def plot_posterior(ax, model, index):

        x = np.linspace(1-3*target_scale, 1+3*target_scale, 10000)
        print("Gamma scale", model.beta(np.array([index,1])))
        ## Plot sampled posterior distribution
        # samples = np.array([model.sample(np.array([index, 1]), norm.rvs(size=2)) for _ in range(len(x))])
        # sns.kdeplot(samples.reshape(-1), label="Posterior Samples", legend=False, ax=ax)
        ax.set_title("State" + str(index))
        ax.set_xticks(np.linspace(1-3*target_scale, 1+3*target_scale, 3)[1:-1])
        ax.set_xlim(1-3*target_scale, 1+3*target_scale)

        # ax.plot(x, [model.pdf(i, np.array([index])) for i in x], linewidth=2, label="Posterior PDF")
        ax.plot(x, norm(loc=1, scale=model.expected_variance(np.array([index,1]))).pdf(x), linewidth=2, label="Posterior PDF")
        ax.plot(x, final_state_posterior.pdf(x), linewidth=2, label="Target")


    number_of_subplots=states
    fig, axs = plt.subplots(1, number_of_subplots, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.01)

    for i, state in enumerate(range(states)):
        plot_posterior(axs[i], model, i+1)

    #plt.legend(loc='best', frameon=False)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel('State Value')
    plt.ylabel('Probability', labelpad=80)
    plt.show()
    

def main(argv):

    target_scale = FLAGS.scale
    
    model = GaussianRegression2(dim=2)
    n_state_prop(model, FLAGS.states, target_scale)

if __name__ == '__main__':

    app.run(main)
