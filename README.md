# Master Thesis

# Current implementations

_Deep RL implementation found in [my dopamine fork](https://github.com/ramunter/dopamine_mirror)_

## Corridor experiment

`python -um src.experiments.main`

>Flags  
>  --iterations: Number of attempts per chain length (default: '5')  
>  --longest_chain: Longest chain attempted (default: '10')  
>  --plot_name: Plot file name

Runs the corridor experiment using the agents selected in the src/experiments/main.py file.  

## Dataset creator

`python -um src.experiments.create_dataset`

Creates a csv file based on an attempt on the corridor environment. This files contains the state, action, estimate and target value from each step in the training process.

## 2 State Propagation

`python -um src.experiments.2_state_prop`

Runs the Osband experiment of a deterministic chain with two states and a known posterior over the terminal state. Plots the results for the 4 different linear bayesian methods explored in the thesis.

## N State Propagation

`python -um src.experiments.n_state_prop`

Runs the same Osband experiment but with n states to investigate how far the variance propagates. Uses the linear bayesian model with unknown mean and noise variance.

>Flags  
>  --scale: SD of target (default: '1.0')  
>  --states: Number of states (default: '5')
