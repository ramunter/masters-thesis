import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from q_learner import q_learner
from corridor import Corridor

def increasing_chain_length_experiment(
    methods, 
    method_names, 
    chain_length_sequence, 
    attempts_per_chain_length):

    names = ["Chain Length"] + method_names
    data = [chain_length_sequence, ]+ [range(0,len(chain_length_sequence))]*3
    
    results = pd.DataFrame(dict(zip(names, data)))

    for j, name in enumerate(method_names):

        print("Now using:", name)
        method = methods[j]

        for i, N in enumerate(tqdm(results["Chain Length"])):

            env = Corridor(N=N, K=0)
            successes = 0
            
            for _ in trange(0, attempts_per_chain_length):

                learned_optimal_policy = q_learner(env, Critic=method, episodes=1000, verbose=True)
                if learned_optimal_policy:
                    successes += 1

            results.loc[results.index[i], name] = successes/attempts_per_chain_length

    print(results)

    results_melted = pd.melt(results, id_vars=["Chain Length"])

    sns.relplot(x="Chain Length", y="value", hue="variable", kind="line", data=results_melted)
    
    plt.savefig("results_new.png")