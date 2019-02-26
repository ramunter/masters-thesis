import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from collections import namedtuple

plt.style.use('ggplot')

Agent = namedtuple('Agent', ['method', 'critic'])

def experiment(
        environment,
        N_list,
        methods,
        critics,
        attempts_per_N):
    """ Runs a group of agents on an environment with an increasing size.

        Combines all the critics and methods to create a set of agents. Each agent
        gets a set number of attempts on each environment size. The method records 
        and plots the average number of episodes required to learn each environment
        size for each agent.

        args:  
            environment (gym.Env): Test enviroment class.  
            N_list ([int]): List of environment sizes to use.  
            methods ([fn]): List of methods to use (For example, q_learning).  
            critics ([CriticTemplate]): List of critics to use.  
            attempts_per_N (int): Number of attempts per agent on each environment size.
    """
    
    names, agents = create_critics(methods, critics)
    results = create_result_df(names, N_list)

    for j, name in enumerate(names):
        
        print("\n=====Now using:", name, "========")
        agent = agents[j]

        for i, N in enumerate(tqdm(results["N"])):

            env = environment(N=N, K=0, p=1)
            results.loc[results.index[i], name] = run_all_attempts(env, agent, N, attempts_per_N)

    print(results)
    plot_results(results)

def run_all_attempts(env, agent, N, attempts):

    sum_steps_to_learn = 0

    for _ in trange(0, attempts):

        steps_to_learn = agent.method(env, agent.critic, episodes=1000, verbose=False)
        sum_steps_to_learn += steps_to_learn

    average_steps_to_learn = sum_steps_to_learn/attempts
    return average_steps_to_learn

def create_result_df(names, N_list):
    df_names = ["N"] + names
    data = [N_list, ] + [range(0, len(N_list))]*len(names) 
    results = pd.DataFrame(dict(zip(df_names, data)))
    return results

def create_critics(methods, critics):

    names = []
    agents = []

    for critic_name, critic in critics.items():
        for method_name, method in methods.items():
            names.append(method_name + " " + critic_name)
            agents.append(Agent(method, critic))
            
    return names, agents

def plot_results(results):

    results_melted = pd.melt(results, id_vars=["N"], value_name="Episodes to Learn")
    print(results_melted)
    sns.relplot(x="N", y="Episodes to Learn", hue="variable", kind="line", data=results_melted)
    plt.savefig("latest_experiment.png")