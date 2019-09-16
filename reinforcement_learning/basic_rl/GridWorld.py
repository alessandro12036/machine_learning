#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:53:30 2019

@author: alessandro
"""

import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})




class Model:
    
    def __init__(self):
        self.theta = np.random.randn(25) / np.sqrt(25)
    
    def s2sx(self, sa):
        s, a = sa[0], sa[1]
        
        feats = np.array([
                s[0] - 1 if a == "U" else 0,
                s[1] - 1.5 if a == "U" else 0,
                (s[0] * s[1] - 3) / 3 if a == "U" else 0,
                (s[0]**2 - 2) / 2 if a == "U" else 0,
                (s[1]**2 - 4.5) / 4.5 if a == "U" else 0,
                1 if a == "U" else 0,
                
                s[0] - 1 if a == "D" else 0,
                s[1] - 1.5 if a == "D" else 0,
                (s[0] * s[1] - 3) / 3 if a == "D" else 0,
                (s[0]**2 - 2) / 2 if a == "D" else 0,
                (s[1]**2 - 4.5) / 4.5 if a == "D" else 0,
                1 if a == "D" else 0,
                
                s[0] - 1 if a == "R" else 0,
                s[1] - 1.5 if a == "R" else 0,
                (s[0] * s[1] - 3) / 3 if a == "R" else 0,
                (s[0]**2 - 2) / 2 if a == "R" else 0,
                (s[1]**2 - 4.5) / 4.5 if a == "R" else 0,
                1 if a == "R" else 0,
                
                s[0] - 1 if a == "L" else 0,
                s[1] - 1.5 if a == "L" else 0,
                (s[0] * s[1] - 3) / 3 if a == "L" else 0,
                (s[0]**2 - 2) / 2 if a == "L" else 0,
                (s[1]**2 - 4.5) / 4.5 if a == "L" else 0,
                1 if a == "L" else 0,
                
                1])
    
        return feats
            
        #return np.array([1, s[0] - 1, s[1] -1.5, s[0]*s[1]-3])
        
    def predict(self, s, a):
        sx = self.s2sx((s, a))
        pred = self.theta.dot(sx)
        return pred


class Grid:
    
    def __init__(self, height, width, standard=True, start=(1,0)):
        
        self.start = start
        self.i = start[0]
        self.j = start[1]
        self.width = width
        self.height = height
        self.V = {}
        self.Q = {}
        self.V_gd = {}
        self.policy = {}
        self.possible_actions = ["U", "D", "R", "L"]
        
        if standard:
            self.standardGrid()            
            
        
    def setActionsRewards(self, actions, rewards):  # Dictionaries
        
        self.actions = actions
        self.rewards = rewards
        
        
    def setMap(self, boulders=[]):  # creates an array of zeros for the map. The boulder(s) are mapped with -1, everything
                                    # else is derived from the class attributes 
        
        self.map_layout = np.zeros((self.height, self.width))
        for boulder in boulders:
            i, j = boulder
            self.map_layout[i, j] = -1 # sets boulders as -1
            
        
    def getCurrentState(self):
        
        return (self.i, self.j)
    
        
    def standardGrid(self):
        
        self.setMap(boulders=[(1, 1)])
        
        actions = {(0,0): ["R", "D"],
                   (0,1): ["L", "R"],
                   (0,2): ["L", "D", "R"],
                   (1,0): ["U", "D"],
                   (1,2): ["R", "U", "D"], 
                   (2,0): ["U", "R"],
                   (2,1): ["L", "R"],
                   (2,2): ["L", "U", "R"],
                   (2,3): ["L", "U"]}
        
        rewards = {(0, 3): 1,
                   (1, 3): -1}
        
        self.setActionsRewards(actions, rewards)
        
    
    def getAllStates(self):
        
        s = list(self.actions.keys()) + list(self.rewards.keys())
        return s
        
        
    def move(self, action):
        
        if action in self.actions[self.getCurrentPos()]:
        
            if action == "R":
                self.j += 1
            elif action == "L":
                self.j -= 1
            elif action == "U":
                self.i -= 1
            elif action == "D":
                self.i += 1
                
            else:
                print("{} is not a valid action".format(action)) # only relevant when player is human
        
        if (self.i, self.j) in self.rewards.keys():
            return self.rewards[(self.i, self.j)]
        else:
            return -0.1
        
    
    def setPosition(self, s):
        
        self.i, self.j = s
        
            
    def unmove(self, prev_action):
        
        if prev_action == "R":
            self.j -= 1
        elif prev_action == "L":
            self.j += 1
        elif prev_action == "U":
            self.i += 1
        elif prev_action == "D":
            self.i -= 1
        else:
            print("{} is not a valid action".format(prev_action))  # only relevant when player is human
            
            
    def gameOver(self, verbose=False):
        
        isOver = (self.i, self.j) not in self.actions.keys()
        
        if isOver and verbose:
            
            if self.rewards[(self.i, self.j)] == 1:
                print("You win!")
            else:
                print("You lose!")
            
        return isOver
    
    
    def getCurrentPos(self):
        return (self.i, self.j)
    
    
    def showGrid(self):  # Draws the board
        cell_space = 10
        for i in range(self.height):
            print("-" * ((self.width * cell_space + 5)))
            print(("|" + " " * cell_space) * (self.width+1))
            print(("|" + " " * cell_space) * (self.width+1))
            
            for j in range(self.width):
                
                print("|", end="")
                v = self.map_layout[i, j]
                                                
                if (i, j) in self.rewards.keys():
   
                    if self.rewards[i, j] == 1:
                        print(" " * 4 + "W" + " " * 5, end="")
                        
                    elif self.rewards[i, j] == -1:
                        print(" " * 4 + "L" + " " * 5, end="")
                
                elif i == self.i and j == self.j:
                    print(" " * 4 + "O" + " " * 5, end="")
                
                elif v == -1:
                    print(" " * 4 + "X" + " " * 5, end="")
                
                else:
                    print(" " * (cell_space), end="")
                
            print("|")
            print(("|" + " " * cell_space) * (self.width+1))
            print(("|" + " " * cell_space) * (self.width+1))
        
        print("-" * ((self.width * cell_space + 5)))
        
         
    def train(self, gamma = 0.9, threshold=0.001, verbose=False, algorithm="sarsaWithGD", s_prob=1.0, N=10000):
        
        states = self.getAllStates()
                
        for s in states:        # initialize values
                self.V[s] = 0.0
        
        if algorithm == "montecarlo":
            self.montecarlo(gamma, s_prob, N)
            
        elif algorithm == "t0":
            self.t0()
            
        elif algorithm == "sarsa":
            self.sarsa()
            
        elif algorithm == "Qlearning":
            self.Qlearning()
            
        elif algorithm == "montecarloWithGD":
            self.montecarloWithGD()
            
        elif algorithm == "sarsaWithGD":
            self.sarsaWithGD()
        
        else:
            self.policyEvaluation(gamma, threshold, s_prob)
            
    
    def random_action(self, a, p=0.1):
        
        if np.random.random() < (1-p):
            return a
        
        else:
#            return np.random.choice(self.possible_actions)
            tmp = list(self.possible_actions)
            tmp.remove(a)
            idx = np.random.randint(len(tmp))
            return tmp[idx]
        
     
    
    
    def Qlearning(self, gamma=0.9, alpha=0.1, N=10000):
        
        update_counter = {}
        
        for s in self.actions.keys():
            self.policy[s] = np.random.choice(self.possible_actions)
        
        for s in self.getAllStates():
            self.Q[s] = {}
            update_counter[s] = {}
            for a in self.possible_actions:
                self.Q[s][a] = 0
                update_counter[s][a] = 1.0
        
        
        for i in range(N):
            
            s1 = (2,0)
            self.setPosition(s1)
            a1 = np.random.choice(self.possible_actions)
            
            while self.gameOver() == False:
                
                r = self.move(a1)
                s2 = self.getCurrentPos()
                a2 = np.random.choice(self.possible_actions)
                
                tmp_alpha = alpha / update_counter[s1][a1]
                update_counter[s1][a1] += 0.005
                if a2:
                    second_term = gamma * self.maxDict(self.Q[s2])[0]
                else:
                    second_term = 0
                self.Q[s1][a1] = self.Q[s1][a1] + tmp_alpha * (r + second_term - self.Q[s1][a1])
                
                s1 = s2
                a1 = a2
        
        for s in self.actions.keys():
            
            best_v, best_p = self.maxDict(self.Q[s])
            self.policy[s] = best_p
            self.V[s] = best_v
    
    
    def sarsa(self, gamma=0.9, epsilon=0.1, alpha=0.1, N=10000):
        
        update_counter = {}
        
        for s in self.actions.keys():
            self.policy[s] = np.random.choice(self.possible_actions)
        
        for s in self.getAllStates():
            self.Q[s] = {}
            update_counter[s] = {}
            for a in self.possible_actions:
                self.Q[s][a] = 0
                update_counter[s][a] = 1.0
        
        t = 1
        
        for i in range(N):
            
            if i % 100 == 0:
                t += 10e-3
            s1 = (2,0)
            self.setPosition(s1)
            a1 = self.random_action(self.maxDict(self.Q[s1])[1], 0.5/t)
            
            while self.gameOver() == False:
                
                r = self.move(a1)
                s2 = self.getCurrentPos()
                a2 = self.random_action(self.maxDict(self.Q[s2])[1], 0.5/t)
                
                tmp_alpha = alpha / update_counter[s1][a1]
                update_counter[s1][a1] += 0.005
                if a2:
                    second_term = gamma * self.Q[s2][a2]
                else:
                    second_term = 0
                self.Q[s1][a1] = self.Q[s1][a1] + tmp_alpha * (r + second_term - self.Q[s1][a1])
                
                s1 = s2
                a1 = a2
        
        for s in self.actions.keys():
            
            best_v, best_p = self.maxDict(self.Q[s])
            self.policy[s] = best_p
            self.V[s] = best_v
            
    
    def sarsaWithGD(self, gamma=0.9, epsilon=0.1, lr=0.01, N=20000):
        
        model = Model()
        t1 = 1
        t2 = 1
        for i in range(N):
            
            if i % 100 == 0:
                t1 += 10e-3
                t2 += 0.01
                
            alpha = lr/t2
            s1 = (2,0)
            self.setPosition(s1)
            qs1 = self.getQS(model, s1)
            a1 = self.maxDict(qs1)[1]
            a1 = self.random_action(a1, 0.5/t1)
            
            while self.gameOver() == False:
                
                r = self.move(a1)
                s2 = self.getCurrentPos()
                
                if s2 in self.rewards.keys():
                    target = r
                else:
                    qs2 = self.getQS(model, s2)
                    a2 = self.maxDict(qs2)[1]
                    a2 = self.random_action(a2, 0.5/t1)
                    target = r + gamma * model.predict(s2, a2)
                
                model.theta += alpha * (target - model.predict(s1, a1)) * model.s2sx((s1, a1))
                s1 = s2
                a1 = a2
                
        for s in self.actions.keys():
            self.Q[s] = self.getQS(model, s)
            self.policy[s] = self.maxDict(self.Q[s])[1]
                
        
    def montecarlo(self, gamma, s_prob, N):
                
        states = list(self.actions.keys())
        
        returns = {}
        
        for s in self.getAllStates():
            self.Q[s] = {}
            returns[s] = {}
            for a in self.possible_actions:
                self.Q[s][a] = 0
                returns[s][a] = []
            
        for s in states:
            idx = np.random.randint(len(self.possible_actions))
            self.policy[s] = self.possible_actions[idx] 
        
        for i in range(N):
            
            s = (2,0)
            self.setPosition(s)
            a = self.random_action(self.policy[s], 0.2)
            
            states_and_rewards = [[s, a, 0],]
            states_and_returns = [] 
            
            while True:

                action = self.random_action(a, 0)
                r = self.move(action)
                s = self.getCurrentPos()
                
                if self.gameOver():
                    states_and_rewards.append((s, None, r))
                    break
                else:
                    a = self.random_action(self.policy[s], 0.2)
                    states_and_rewards.append((s, a, r))
            
            g = 0
            first = True
            
            for s, a, r in reversed(states_and_rewards):
                
                if first == True:
                    first = False
                else:
                    states_and_returns.append((s, a, g))
                g = r + (gamma * g)
            
            states_and_returns = reversed(states_and_returns)
            
            seen_states = set()
            
            for s, a, g in states_and_returns:
                
                if (s, a) not in seen_states:
                    returns[s][a].append(g)
                    self.Q[s][a] = np.mean(returns[s][a])
                    seen_states.add((s,a))

                        
            for s in self.actions.keys():
                
                best_p = self.maxDict(self.Q[s])[1]
                self.policy[s] = best_p
    
    
    def montecarloWithGD(self, gamma=0.9, alpha=0.001, n=20000):
        
        self.V_gd = {}
        returns = {}
        
        for s in self.getAllStates():
            returns[s] = []
            
        self.policy = {(0,0): "R",
               (0,1): "R",
               (0,2): "R",
               (1,0): "U",
               (1,2): "U",
               (2,0): "U",
               (2,1): "L",
               (2,2): "U",
               (2,3): "L"}
            
        theta = np.random.randn(4) / 2
        t = 1
        
        for i in range(n):
            if i % 100 == 0:
                t += 0.01
            
            tmp_alpha = alpha/t
            states_and_returns = []
            idx = np.random.randint(len(self.actions.keys()))
            s = list(self.actions.keys())[idx]
            states_and_rewards = [(s,0)]
            self.setPosition(s)
        
            while self.gameOver() == False:
                
                action = self.policy[s]
                a = self.random_action(action, 0.5)
                r = self.move(a)
                s = self.getCurrentPos()
                states_and_rewards.append((s, r))
            
            g = 0
            first = True
            for s, r in reversed(states_and_rewards): # sets the ground truth? 
                if first:
                    first=False
                else:
                    states_and_returns.append((s, g))
                g = r + gamma*g
            
            states_and_returns = reversed(states_and_returns)
            seen_states = []
            for s, g in states_and_returns:
                
                if s not in seen_states:
                    
                    returns[s].append(g)
                    
                    sx = self.create_features(s)
                    v_pred = theta.dot(sx)
                    theta = theta + (tmp_alpha * (g-v_pred)) * sx
                    seen_states.append(s)
            
        for s in self.actions.keys():
            
            self.V[s] = np.mean(returns[s])
            
            sx = self.create_features(s)
            self.V_gd[s] = theta.dot(sx)
            
            
    def create_features(self, s):
        
        sx = np.array([1, s[0] - 1, s[1] -1.5, s[0]*s[1]-3])
        return sx
    
    
    def t0(self, N=200000, lr=0.1, gamma=0.9):
        
        model = Model()
        
        self.policy = {
                        (2, 0): 'U',
                        (1, 0): 'U',
                        (0, 0): 'R',
                        (0, 1): 'R',
                        (0, 2): 'R',
                        (1, 2): 'R',
                        (2, 1): 'R',
                        (2, 2): 'R',
                        (2, 3): 'U',
                      }
        k = 1.0
        for i in range(N):
            alpha = lr/k
            if i % 10 == 0:
                k += 0.01
            
#            s = (2,0)
            idx = np.random.randint(len(self.actions.keys()))
            s = list(self.actions.keys())[idx]
            self.setPosition(s)
            states_and_rewards = [(s, 0)]
            
            while self.gameOver() == False:
                
                a = self.random_action(self.policy[s], 0)
                r = self.move(a)
                s = self.getCurrentPos()
                states_and_rewards.append((s, r))
                        
            for t in range(len(states_and_rewards)-1):
                st, _ = states_and_rewards[t]
                st1, r = states_and_rewards[t+1]
                
                if st1 in self.rewards.keys():
                    target = r
                else:
                    target = r + gamma*model.predict(st1)
                
                model.theta = model.theta + alpha * (target - model.predict(st)) * model.s2sx(st)
                self.V[st] = self.V[st] + alpha * (r + gamma*self.V[st1] - self.V[st])
            
            
        for s in self.actions.keys():
            self.V_gd[s] = model.predict(s)
            
    def maxDict(self, d):
        
        best_value = float("-inf")
        best_policy = None
        for a in self.possible_actions:
            v = d[a]
            if v > best_value:
                best_value = v
                best_policy = a
        return best_value, best_policy
    
    
    def policyEvaluation(self, gamma, threshold, s_prob):  # Commented out a more straightforward albeit slower method
        
        n_iterations = 0
        states = self.getAllStates()
        ps = s_prob # probability that by doing action a we'll end up in state s'
        
        
#        for s in states:        # initialize policies randomly
#            c = np.random.randint(0, len(self.possible_actions))
#            self.policy[s] = self.possible_actions[c]
        

        while True:
            
            n_iterations += 1            
            
                
            biggest_delta = 0
            
            
            for s in states:
                
                
                best_value = float("-inf")
                old_v = self.V[s]
                if s in self.actions.keys():
                    for policy in self.possible_actions:
                        self.setPosition(s)
                        new_v = 0
                        for action in self.possible_actions:
                            self.setPosition(s)
                            if action == policy:
                                p = ps
                            else:
                                p = (1-ps) / 3
                            r = self.move(action)
                            new_s = self.getCurrentPos()
                            new_v += p * (r + gamma*self.V[new_s])
                        if new_v > best_value:
                            best_value = new_v
                    self.V[s] = best_value
                    delta = abs(old_v - best_value)
                    if delta > biggest_delta:
                        biggest_delta = delta
            if biggest_delta < threshold:
                break
            
   
#            #policy iteration
#            while True:
#                
#                biggest_delta = 0
#                for s in states:
#                    if s in self.actions.keys():
#                        new_v = 0
#                        old_v = self.V[s]
#                        for action in self.possible_actions:
#                            self.setPosition(s)
#                            if action == self.policy[s]:
#                                p = ps
#                            else:
#                                p = (1-ps) / 3
#                            #action = self.policy[s]
#                            r = grid.move(action)
#                            new_s = self.getCurrentPos()
#                            new_v += p * (r + gamma*self.V[new_s])
#                            
#                        delta = abs(new_v - old_v)
#                        self.V[s] = new_v
#                        if delta > biggest_delta:
#                            biggest_delta = delta
#                        
#
#                if biggest_delta < threshold:
#                    break
#                    
#            #policy evaluation
#                
#            policy_is_converged = True
#            
#            for s in states:
#                if s in self.actions.keys():
#                    best_policy = None
#                    old_policy = self.policy[s]
#                    best_value = float("-inf")
#                    for a in self.possible_actions:
#                        new_v = 0
#                        temp_policy = a
#                        for action in self.possible_actions:
#                            self.setPosition(s)
#                            if action == temp_policy:
#                                p = ps
#                            else:
#                                p = (1-ps) / 3
#                            r = self.move(action)
#                            new_s = self.getCurrentPos()
#                            if new_s in states:
#                                new_v += p * (r + gamma * (self.V[new_s]))
#                        if new_v > best_value:
#                            best_value = new_v
#                            best_policy = temp_policy
#                            
#                    self.policy[s] = best_policy    
#                    if best_policy != old_policy:
#                        policy_is_converged = False
#                        
#            if verbose and iteration_n % 10 == 0:
#                    print(str(iteration_n) + ": ", end="")
#                    self.showValues()
#                    
#            if policy_is_converged:
#                break
        
                
    def showValues(self, form="Q"):
        
        value_map = np.zeros_like(self.map_layout)
        
        for s in self.actions.keys():
            if form=="Q":
                value_map[s[0], s[1]] = self.Q[s][self.policy[s]]
            elif form=="V_gd":
                value_map[s[0], s[1]] = self.V_gd[s]
            else:
                value_map[s[0], s[1]] = self.V[s]
            
        print(value_map)
        
        
    def reset(self):
        
        self.i, self.j = self.start
    
    
    def showPolicy(self):
        
        for r in range(3):
            for c in range(4):
                if (r, c) in self.policy.keys():
                    v = self.policy[(r,c)]
                else:
                    v = " "
                print(v, end="    ")
            print()
            
            
    def getQS(self, model, s):
    
        Qs = {}
        for a in self.possible_actions:
            pred = model.predict(s, a)
            Qs[a] = pred
        return Qs
        

def playGame(grid):
    
    while True:
        
        grid.showGrid()
        
        action = input("Enter move: ")
        action = action.upper()
                
        if action == "Q":
            break
            
        s = grid.getCurrentState()
        
        if action in grid.actions[s]:
            r=grid.move(action)
        else:
            print("Move not allowed")
        if grid.gameOver():
            break
        
if __name__ == "__main__":
    
    grid = Grid(3, 4)
    grid.train(verbose=True, s_prob=1)
    grid.showValues(form="Q")
    #grid.showValues(form="")
    
    grid.showPolicy()