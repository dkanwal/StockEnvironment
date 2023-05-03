from matplotlib.pyplot import flag
import numpy as np
import random as random


class TabularQLearner:
    def __init__(self, states = 100, actions = 4, alpha = 0.2, gamma = 0.9, epsilon = 0.98, epsilon_decay = 0.999, dyna = 0):
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.Q = np.random.random((states, actions))
        self.s = None
        self.a = None
        self.dyna = dyna
        self.d_list = []

    def train(self, s, r):
        rand_num = np.random.random()
        state_row = self.Q[s]
        a = np.argmax(state_row)
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha*(r + self.gamma * self.Q[s, a]) 
        if self.dyna != 0:
            self.d_list.append([self.s, self.a, s, r])
            for i in range(0, self.dyna):
                selection = random.randint(0, len(self.d_list) - 1)
                sel_tuple = self.d_list[selection]
                sel_state = sel_tuple[2]
                self.Q[sel_tuple[0], sel_tuple[1]] = (1 - self.alpha) * self.Q[sel_tuple[0], sel_tuple[1]] + self.alpha*(sel_tuple[3] + self.gamma * np.max(self.Q[sel_state]))

            if rand_num < self.epsilon:
                a = random.randint(0, self.actions - 1)
                self.epsilon = self.epsilon * self.epsilon_decay

        self.s = s
        self.a = a   

        return a 
    
    def test(self, s):
        state_row = self.Q[s]
        a = np.argmax(state_row)
        self.s = s
        self.a = a
        return a

    
