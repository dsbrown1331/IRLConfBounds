'''
Created on Jan 23, 2017

@author: dsbrown
'''

import numpy as np
from solvers import mdp
import math

#TODO figure out terminal and None action with end of game

class MountainCarMDP(mdp.MDP):
    """continuous mountain car mdp"""


    def __init__(self, gamma=.95):
        self.gamma = gamma
        self.min_position = -1.2
        self.max_position = 0.6
        self.min_speed = -0.07
        self.max_speed = 0.07
        self.goal_position = 0.5
        
        self.action_list = [0,1,2]
       
    
    #mountain car always has reward of -1
    def R(self, state):
        return -1;  
    
    def actions(self, state):
        if state[0] > self.goal_position:
            return [None]
        else:
            return self.action_list
    #transitions use math from OpenAIGym and then discretize the resulting position and value
    #actions are LEFT: 0, COAST: 1, RIGHT: 2 
    def T(self, state, action):
        if action == None:
            return [(0.0, state)]
        else:
            position, velocity = state
            #run action for ten steps
            for _ in range(4):
                velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
                velocity = np.clip(velocity, -self.max_speed, self.max_speed)
                position += velocity
                position = np.clip(position, self.min_position, self.max_position)
            if (position==self.min_position and velocity<0): velocity = 0
            #discretize the resulting position
            next_state = (position,velocity)
            return [(1.0, next_state)]

    
    

