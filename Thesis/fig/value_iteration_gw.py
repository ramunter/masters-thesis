import numpy as np


def update_best_value(row, col, dim, best_value, value_grid):

    value = 0.9*value_grid[row, col]

    if value > best_value:
        best_value = value 
    return best_value

def get_best_action(state, dim, value_grid):
    state = state
    col = state % dim
    row = state / dim
    
    value_grid = np.reshape(value_grid, (dim, dim))
    
    if row == (dim-1) and col == (dim-1):
        best_value = 10
        return best_value
    
    else:
        reward = -1

    best_value = -10000
    if col + 1 < dim:
        best_value = update_best_value(row, col+1, dim, best_value, value_grid)

    if col - 1 >= 0:
        best_value = update_best_value(row, col-1, dim, best_value, value_grid)
    
    if row + 1 < dim:
        best_value = update_best_value(row+1, col, dim, best_value, value_grid)
    
    if row - 1 >= 0:
        best_value = update_best_value(row-1, col, dim, best_value, value_grid)

    best_value = reward + best_value
    return best_value


def value_iteration(dim, final_delta=0.01):
    value_grid = np.zeros(dim*dim)
    value_grid[15] = 10

    delta = final_delta + 1
    while delta > final_delta:
        delta = 0
        for state in range(0, dim*dim):
            v = value_grid[state]
            value_grid[state] = get_best_action(state, dim, value_grid)
            delta = max(delta, abs(v-value_grid[state]))

        print(np.reshape(value_grid, (4, 4)))

    return value_grid

if __name__ == '__main__':
    value_grid = value_iteration(4)
    print(np.reshape(value_grid,(4,4)))
