'''
Created on Jan 26, 2017

@author: dsbrown
'''
#TODO could check out shaping rewards using BIRL and RHC and RBF rewards to see how it affects things...

#doesn't work since recursion is too slow! It's too slow to do full horizon since states are never repeated. 

from domains import mountain_car_discrete as mcd
from domains import mountain_car_continuous as mcc
import matplotlib.pyplot as plt
from solvers.mdp import policy_iteration
from solvers import rhc
import random

def main():
    
    #first solve using a discretized version of mountain car
    world = mcd.MountainCarGridMDP(20)
    
    opt_pi, opt_v = policy_iteration(world)
    #print opt_pi
    #print opt_v
    #print opt_pi.values()
    #start at bottom of hill
    next_state = world.convert_to_discrete((-0.5,0))
    pos_history = []
    vel_history = []
    p,v = next_state
    pos_history.append(p)
    vel_history.append(v)
    for i in range(25):
        next_states = world.T(next_state, opt_pi[next_state])
        #print "taking action", opt_pi[next_state]
        prob, next_state = next_states[0]
        p,v = next_state
        pos_history.append(p)
        vel_history.append(v)
        #if p > world.goal_position:
        #    print "finished"
     
        #print prob,next_state
    #print pos_history
    #print vel_history
    plt.plot(pos_history,vel_history)
    plt.plot(pos_history[0],vel_history[0],'bo')
    plt.plot(pos_history[-1],vel_history[-1],'r^')
    plt.show()
    
    #now solve a continuous version using RHC
    world_cont = mcc.MountainCarMDP()
    horizon = 5
    print rhc.compute_qvals((0.3,0.02), horizon, world_cont)


if __name__ == '__main__':
    main()