import util
import numpy as np

from math import exp

capacities, weights, profits, solution = util.get_data('3')
knapsacks = weights.shape[0]
objects = weights.shape[1]

def is_valid(state):
    knapsack_weights = np.multiply(state, weights).sum(axis=1)
    return np.all(knapsack_weights <= capacities)

def profit(state):
    return np.sum(np.multiply(profits,state))

def acceptance(profit_new, profit_current, T, k=0.1):
    return exp((profit_new - profit_current)/k/T)

def cooling(T, T_start, step, total_steps):
    return T_start * (1 - step/total_steps)
    
def neighbour(state):
    state_copy = state.copy()
    random_object = np.random.randint(objects)

    state_copy[:,random_object] = 0
    if np.random.rand() < 0.9:
        random_knapsack = np.random.randint(knapsacks)
        state_copy[random_knapsack, random_object] = 1
    
    return state_copy

def generate_random_state():
    while True:
        object_indices = np.random.randint(0, knapsacks, objects)
        state = np.zeros(weights.shape, dtype=int)

        state[object_indices, np.arange(objects)] = 1

        if is_valid(state):
            return state


class SimulatedAnnealingKnapsack:
    def __init__(
            self,
            neighbour=neighbour,
            profit=profit,
            acceptance=acceptance,
            cooling=cooling,
            T_start=100.0,
            total_steps=5000):
        self.state_current = np.zeros(weights.shape)#_generate_random_state()
        self.profit_current = 0

        self.state_best = None
        self.profit_best = 0

        self.neighbour = neighbour
        self.profit = profit
        self.acceptance = acceptance
        self.cooling = cooling
        self.T_start = T_start
        self.total_steps = total_steps
    
    def iteration(self, T):
        state_new = self.neighbour(self.state_current)
        if not is_valid(state_new):
            return

        profit_new = self.profit(state_new)

        dE = self.profit_current - profit_new

        if dE < 0:
            accept = True
        else:
            accept = np.random.rand() < self.acceptance(profit_new, self.profit_current, T)
        
        if accept:
            self.state_current = state_new
            self.profit_current = profit_new
            if profit_new > self.profit_best:
                self.profit_best = profit_new
                self.state_best = state_new
    
    def anneal(self):
        T = self.T_start
        for step in range(self.total_steps+1):
            if step % (self.total_steps/10) == 0:
                print(f'Step - {step}, Temp - {T:.1f}, Profit - {self.profit_current}')
            self.iteration(T)
            T = self.cooling(T, self.T_start, step, self.total_steps)

if __name__ == '__main__':
    optimizer = SimulatedAnnealingKnapsack(total_steps=10000)

    optimizer.anneal()

    print(f'Best positions: {np.argmax(optimizer.state_best, axis=0)+1}')
    print(f'Best profit: {optimizer.profit_best}')

    if solution is not None:
        desired_state = np.zeros(weights.shape, dtype=int)
        desired_state[solution-1, np.arange(objects)] = 1
        desired_profit = profit(desired_state)

        print(f'Desired positions: {solution}')
        print(f'Desired profit: {desired_profit}')
