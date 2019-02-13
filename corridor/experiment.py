import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from q_learner import q_learner
from g_learner import g_learner
from corridor import Corridor

def increasing_chain_length_experiment(
    methods, 
    method_names, 
    chain_length_sequence, 
    attempts_per_chain_length):

    names = ["Chain Length"] + method_names + ["G_"+name for name in method_names]
    data = [chain_length_sequence, ]+ [range(0,len(chain_length_sequence))]*len(method_names)*2
    
    results = pd.DataFrame(dict(zip(names, data)))

    for j, name in enumerate(method_names):
        print("\n=====Now using:", name, "========")
        method = methods[j]

        for i, N in enumerate(tqdm(results["Chain Length"])):

            env = Corridor(N=N, K=0, p=1)
            sum_steps_to_learn = 0
            
            for _ in trange(0, attempts_per_chain_length):

                steps_to_learn = q_learner(env, Critic=method, episodes=1000, verbose=False)
                sum_steps_to_learn += steps_to_learn

            results.loc[results.index[i], name] = sum_steps_to_learn/attempts_per_chain_length

        for i, N in enumerate(tqdm(results["Chain Length"])):

            env = Corridor(N=N, K=0, p=1)
            sum_steps_to_learn = 0
            
            for _ in trange(0, attempts_per_chain_length):

                steps_to_learn = g_learner(env, Critic=method, episodes=1000, verbose=False)
                sum_steps_to_learn += steps_to_learn

            results.loc[results.index[i], "G_"+name] = sum_steps_to_learn/attempts_per_chain_length

        

    print(results)

    results_melted = pd.melt(results, id_vars=["Chain Length"])

    sns.relplot(x="Chain Length", y="value", hue="variable", kind="line", data=results_melted)
    
    plt.savefig("results_new.png")
