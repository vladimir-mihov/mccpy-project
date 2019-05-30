import numpy as np
import os

def get_data(dataset):
    dataset_dir = 'datasets'

    base_dir = os.path.join(dataset_dir, dataset)

    capacities_file = os.path.join(base_dir, 'c.txt')
    weights_file = os.path.join(base_dir, 'w.txt')
    profits_file = os.path.join(base_dir, 'p.txt')
    solution_file = os.path.join(base_dir, 's.txt')

    with open(capacities_file) as f:
        capacities = np.array(f.readlines(), dtype=int)

    with open(weights_file) as f:
        weights = np.array(
            list(map(lambda l: np.array(l.split(), dtype=int),f.readlines()))
        )

    with open(profits_file) as f:
        profits = np.array(
            list(map(lambda l: np.array(l.split(), dtype=int),f.readlines()))
        )

    if os.path.exists(solution_file):
        with open(solution_file) as f:
            solution = np.array(f.readline().split(), dtype=int)
    else:
        solution = None

    return capacities, weights, profits, solution
