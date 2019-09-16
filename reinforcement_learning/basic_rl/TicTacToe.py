#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 21:28:20 2019

@author: alessandro
"""

import numpy as np

class Human:
    
    def __init__(self, sym):
        self.sym = sym
    
    def takeAction(self, env, eps, verbose=False):
        
        input_not_valid = True
        
        while input_not_valid:
            
            env.drawBoard()
            print(env.getBoard())
            
            choice = input("Enter comma separated coordinates (q to quit): ")
            
            if choice == "q":
                return -1
            
            choice = choice.replace(" ", "")
            l_choice = choice.split(",")
                            
            try:
                int_choice = map((lambda x: int(x)), l_choice)
                i, j = int_choice
                if env.getBoard()[i, j] == 0:
                    env.setBoard(i, j, self.sym)
                    input_not_valid = False
            except:
                pass
            
            if input_not_valid:
                print("Coordinates not valid, please try again.")
        
        return 0
        
    def updateValues(self):
        pass
    
    def updateHistory(self, s):
        pass
    
    
class Agent:
    
    def __init__(self, sym, lr):
        self.lr = lr
        self.history = []
        self.values = {}
        self.sym = sym
        
    def resetHistory(self):
        self.history = []
        
    def setValues(self, V):
        self.values = V
        
    def setSym(self, sym):
        self.sym = sym
    
    def takeAction(self, env, eps, verbose=True):
        if verbose:
            score_tab = np.zeros_like(env.getBoard())
        
        bestAction = (0, 0)
        bestValue = 0
        
        r = np.random.rand()
        
        possible_actions = []
        
        length = env.length
        for i in range(length):
            for j in range(length):
                if env.getBoard()[i, j] == 0:
                        possible_actions.append((i, j))
        
        if r < eps:
            idx = np.random.randint(0, len(possible_actions))
            choice = possible_actions[idx]
            
        else:
            for action in possible_actions:
                i, j = action
                env.setBoard(i, j, self.sym)
                s = env.getState()  # looks for hash
                value = self.values[s]
                if verbose:
                    score_tab[i, j] = value
                if value >= bestValue:
                    bestValue = value
                    bestAction = action
                    
                env.setBoard(i, j, 0) #set back value to 0
            choice = bestAction
        i, j = choice
        env.setBoard(i, j, self.sym)
           
        
        if verbose:
            print(score_tab)
            print(possible_actions)
            print("chosen action is {}".format(choice))
        
        return 0  # Allows player to quit mid-game
        
    
    def updateValues(self):
        
        if env.getWinner() == self.sym:
            target = 1.0
        else:
            target = 0.0
                            
        for s in reversed(self.history):
            new_v = self.values[s] + self.lr * (target - self.values[s])
            self.values[s] = new_v
            target = self.values[s]
            
        self.resetHistory() 
        
    def updateHistory(self, s):
        self.history.append(s)
    
    
class Enviroment:
    
    def __init__(self, length):
        
        self.length = length
        self.board = np.zeros((length, length))
        self.x = 1
        self.o = -1
        
    
    def resetBoard(self):
        self.board = np.zeros_like(self.board)
        
    
    def getBoard(self):
        return self.board
        
    def setBoard(self, i, j, sym):
        self.board[i, j] = sym
        
    def getState(self):
        
        k = 3
        n = 0
        l = env.length
        v = 0
        hash_value = 0
        
        for i in range(l):
            for j in range(l):
                v = env.getBoard()[i, j]
                hash_value += (k**n) * v
                n += 1
                
        return hash_value
    
    def getWinner(self):
        
        l = self.length
        x_victory = env.x * 3.0
        o_victory = env.o * 3.0
        
        #rows
        for i in range(l):
            if self.board[i, :].sum() == x_victory:
                return env.x
            elif self.board[i, :].sum() == o_victory:
                return env.o
           
        #cols
            if self.board[:, i].sum() == x_victory:
                return env.x
            elif self.board[:, i].sum() == o_victory:
                return env.o
            
        #diags
        if self.board.trace() == x_victory or np.flip(self.board, 1).trace() == x_victory:
            return env.x
        elif self.board.trace() == o_victory or np.flip(self.board, 1).trace() == o_victory:
            return env.o
        return 0
        
    
    def isEnded(self):
        
        if self.getWinner() != 0:
            return True
        for i in range(self.length):
            for j in range(self.length):
                if self.board[i, j] == 0:
                    return False
        return True
        
        
    def drawBoard(self):
        l = self.length
        padding = " " * (10 // 2)
        for i in range(l):
            print(padding + "-" * 13)
            for j in range(l):
                if self.board[i, j] == env.x:
                    sym = "X"
                elif self.board[i, j] == env.o:
                    sym = "O"
                else:
                    sym = " "
                print(padding + sym, end="")
            print("")
        print(padding + "-" * 13)
        

def getHashAndWinner(env, i=0, j=0):
    
    results = []
    
    for sym in [0, env.x, env.o]:
        env.setBoard(i, j, sym)
        
        if j == 2 and i == 2:
            state = env.getState()
            winner = env.getWinner()
            ended = env.isEnded()
            results.append((state, winner, ended))
        else:
            if j < 2:
                results += getHashAndWinner(env, i, j+1)
            else:
                results += getHashAndWinner(env, i+1, 0)
    return results            
        

def initVX(env, triplets):
    V = {}
    v = 0
    for state, winner, ended in triplets:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


def initVO(env, triplets):
    V = {}
    v = 0
    for state, winner, ended in triplets:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V
            

def train(env, p1, p2, epochs, verbose=False, epsilon=0.1):
    
    for epoch in range(epochs):

        playGame(env, p1, p2, verbose=verbose, epsilon=epsilon)
        
        if epoch % 100 == 0:
            print(epoch, " ", epsilon)
                    
                    
def playGame(env, p1, p2, verbose=True, epsilon = 0.0):
    
    env.resetBoard()
    q = 0
    
    human = type(p1).__name__ == "Human" or type(p2).__name__ == "Human"
        
    if human:
        human_is_p1 = type(p1).__name__ == "Human"
    
        symbols = {1: "X",
                   -1: "O"}
            
        if human_is_p1:
            print("You're player 1 and you're using the {}".format(symbols[p1.sym]))
        else:
            print("You're player 2 and you're using the {}".format(symbols[p2.sym]))
    
    while q == 0:
            
        q = p1.takeAction(env, eps=epsilon, verbose=verbose)
        s = env.getState()
        p1.updateHistory(s)
        if env.isEnded():
            p1.updateValues()
            p2.updateValues()
            break
  
        q = p2.takeAction(env, eps=epsilon, verbose=verbose)
        s = env.getState()
        p2.updateHistory(s)
        if env.isEnded():
            p1.updateValues()
            p2.updateValues()
            break
    
    if verbose:
        winner = env.getWinner() 
        if q == -1:
            print("Match stopped")
        elif winner == 0:
            print("Draw")
        else:
            if winner == -1:
                winner = 2
            print("Winner is player {}".format(winner))
            env.drawBoard()

        
if __name__ == "__main__":
    
    env = Enviroment(3)
    p1 = Agent(env.x, 0.5)
    p2 = Agent(env.o, 0.5)
    vals = getHashAndWinner(env)
    p1.setValues(initVX(env, vals))
    p2.setValues(initVO(env, vals))
    train(env, p1, p2, 10001, epsilon=0.1)
    
#    Trains in the opposite roles
#    p1.setSym(env.o)
#    p2.setSym(env.x)
#    train(env, p2, p1, 10001, epsilon=0.1)
    
    
    player = Human(env.o)
    
    while True:
        playGame(env, p1, player)
            
        play_again = input("Do you want to play again?(y/n): ")
        if play_again == "n":
            break