import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from statistics import median, mean
from collections import namedtuple

plt.style.use('ggplot')

Agent = namedtuple('Agent', ['method', 'critic'])


def experiment(
        environment,
        N_list,
        methods,
        critics,
        attempts_per_N,
        filename):
    """
    Runs a group of agents on an environment with an increasing size.

    Combines all the critics and methods to create a set of agents. Each agent
    gets a set number of attempts on each environment size. The method records 
    and plots the average number of episodes required to learn each environment
    size for each agent.

    args:  
        environment    : Test environment class.  
        N_list         : List of environment sizes to use.  
        methods        : List of methods to use (For example, q_learning).  
        critics        : List of critics to use.  
        attempts_per_N : Number of attempts per agent on each environment size.  
        filename       : Name of plot of results. If None will not plot.
    """

    names, agents = create_critics(methods, critics)
    results = create_result_df(names, N_list)

    for j, name in enumerate(names):

        print("\n=====Now using:", name, "========")
        agent = agents[j]

        for i, N in enumerate(tqdm(results["N"])):

            env = environment(N=N)
            results.loc[results.index[i], name] = run_all_attempts(
                env, agent, N, attempts_per_N)

    print(results)
    if filename is not None:
        plot_results(results, filename)


def run_all_attempts(env, agent, N, attempts):
    """
    Runs a certain number of attempts on an environment of size N using the
    given agent.

    args:  
        env      : Test environment class.  
        agent    : Agent to use.  
        N        : Size of environment.  
        attempts : Number of attempts.  
    """
    list_steps_to_learn = []

    for _ in range(0, attempts):

        steps_to_learn = agent.method(
            env, agent.critic, episodes=4000, verbose=False)
        list_steps_to_learn += [steps_to_learn]

    average_steps_to_learn = mean(list_steps_to_learn)
    return average_steps_to_learn


def create_result_df(names, N_list):
    """
    Creates a dataframe for recording the average learning time for the
    different agents.

    args:  
        names : Describing name of each agent.  
        N_list : List of environment sizes to be tested on.  
    returns:  
        results : Empty dataframe for storing results
    """

    df_names = ["N"] + names
    data = [N_list, ] + [range(0, len(N_list))]*len(names)
    results = pd.DataFrame(dict(zip(df_names, data)))
    return results


def create_critics(methods, critics):
    """
    Creates every combination of methods and critics given.

    args:  
        methods : List of methods.  
        critics : List of critics.  

    returns:   
        names : List of describing names for each agent
        agents : List of agents.
    """

    names = []
    agents = []

    for critic_name, critic in critics.items():
        for method_name, method in methods.items():
            names.append(method_name + " " + critic_name)
            agents.append(Agent(method, critic))

    return names, agents


def plot_results(results, filename):
    """
    Plots the average learning time for each agent in each environment.

    args:  
        results : Dataframe containing experiment results.
    """

    results_melted = pd.melt(
        results, id_vars=["N"], value_name="Episodes to Learn")

    results_melted.to_csv("./experiment_results.csv")

    sns.relplot(x="N", y="Episodes to Learn", hue="variable",
                kind="line", data=results_melted)
    plt.savefig(filename+".png")
