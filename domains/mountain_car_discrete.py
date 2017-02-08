'''
Created on Jan 23, 2017

@author: dsbrown
'''

import numpy as np
from solvers import mdp
import math

#TODO figure out terminal and None action with end of game

class MountainCarGridMDP(mdp.MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1].  All you have to do is 
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state).  Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""


    def __init__(self, grid_size, gamma=.95):
        self.gamma = gamma
        self.min_position = -1.2
        self.max_position = 0.6
        self.min_speed = -0.07
        self.max_speed = 0.07
        self.goal_position = 0.5
        discretization = grid_size
        self.pos_discretized = np.linspace(self.min_position, self.max_position, discretization)
        self.vel_discretized = np.linspace(self.min_speed, self.max_speed, discretization)
        self.states = [(p,v) for p in self.pos_discretized for v in self.vel_discretized]
        self.action_list = [0,1,2]
        #print "grid positions", self.pos_discretized
        #print "grid velocities", self.vel_discretized
    
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
            next_disc_state = self.convert_to_discrete((position,velocity))
            return [(1.0, next_disc_state)]

    def convert_to_discrete(self, cont_state):
        """
        takes continuous state and converts to discretization for tabular policy lookup
        """ 
        pos, vel = cont_state
        #search over positions
        pos_idx = self.find_closest_index(pos, self.pos_discretized)
        vel_idx = self.find_closest_index(vel, self.vel_discretized)
        
        return (self.pos_discretized[pos_idx], self.vel_discretized[vel_idx]) 
        
    def find_closest_index(self, val, val_list):
        pos_idx = 0
        while val > val_list[pos_idx]:
            pos_idx += 1
        #check if we should use pos_idx or pos_idx-1, use whichever is closer
        if abs(val_list[pos_idx - 1] - val) < abs(val_list[pos_idx] - val):
            pos_idx -= 1
        return pos_idx
# def main():
#     pos_min = -1.2
#     pos_max = 0.6
#     vel_min = -0.07
#     vel_max = 0.07
#     discretization = 10
#     pos_discretized = np.linspace(pos_min, pos_max, discretization)
#     vel_discretized = np.linspace(vel_min, vel_max, discretization)
#     states = [(p,v) for p in pos_discretized for v in vel_discretized]
#     print pos_discretized
#     print vel_discretized
#     disc_state =  convert_to_discrete((0.16, -0.06), pos_discretized, vel_discretized)
#     print "discretized", disc_state
#     print disc_state in states
    

