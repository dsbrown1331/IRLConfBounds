'''
Created on Jan 26, 2017

@author: dsbrown
'''

def main():
    import solvers.mdp
    import domains.mountain_car_discrete as mc
    import matplotlib.pyplot as plt
    from solvers.mdp import policy_iteration
    import random
    
    world = mc.MountainCarGridMDP(20, gamma=.95)
    
    opt_pi, opt_v = policy_iteration(world)
    print opt_pi
    print opt_v
    print opt_pi.values()
    #start at bottom of hill
    next_state = world.convert_to_discrete((-0.5,0))
    pos_history = []
    vel_history = []
    p,v = next_state
    pos_history.append(p)
    vel_history.append(v)
    for i in range(25):
        next_states = world.T(next_state, opt_pi[next_state])
        print "taking action", opt_pi[next_state]
        prob, next_state = next_states[0]
        p,v = next_state
        pos_history.append(p)
        vel_history.append(v)
     
        print prob,next_state
    print pos_history
    print vel_history
    plt.plot(pos_history,vel_history)
    plt.plot(pos_history[0],vel_history[0],'bo')
    plt.plot(pos_history[-1],vel_history[-1],'r^')
    plt.show()


if __name__ == '__main__':
    main()