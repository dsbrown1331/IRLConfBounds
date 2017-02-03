from solvers import mdp
from birl import feature_birl
from birl import birl_util
from bounds import confidence_bounds
import random
import cPickle as pickle
from navgrid_test_plot import *


##MDP PARAMS
r_min = -1
r_max = 1
size = 15
grid = [[0 for _ in range(size)] for _ in range(size)]
gamma = 0.95

####make random world
init_states = [(0,0),(0,14),(13,0),(14,14)]
term_states = [(7,7)]
weights = [0,-1,1,0,0,-1]
#Features are White, Red, Blue, Yellow, Green, Purple
white_f = [1,0,0,0,0,0]
red_f = [0,1,0,0,0,0]
blue_f = [0,0,1,0,0,0]
yellow_f = [0,0,0,1,0,0]
green_f = [0,0,0,0,1,0]
purple_f = [0,0,0,0,0,1]
non_term_colors = [white_f, red_f, yellow_f, green_f, purple_f,white_f, white_f]
#generate random features for each state
#features = {}
#for row in range(size):
#    for col in range(size):
#        #pick random color for non_terminals
#        rand_color = random.choice(non_term_colors)
#        features[(col,row)] = rand_color
##add terminal state
#for term in term_states:
#    features[term] = blue_f
#pickle.dump(features, open( "grid_nav_world15x15.p", "wb" ) )
features = pickle.load( open( "../grid_nav_world15x15.p", "rb" ) )
print features


true_mdp = mdp.DeterministicWeightGridMDP(features,
                weights, grid, terminals=term_states, init=init_states,
                gamma=gamma,r_min=r_min,r_max=r_max)
true_mdp.print_rewards()
true_mdp.print_arrows()


## use hard coded demos from optimal policy
opt_pi, opt_U = mdp.policy_iteration(true_mdp)
demos_all = []
print true_mdp.init
for s0 in true_mdp.init:
    d = []
    state = s0
    print "state", state
    while state not in true_mdp.terminals:
       action = opt_pi[state]
       print "action", action
       d.append((state, action))
       print "demos", d
       state = true_mdp.go(state,action)
       print "next state", state
    d.append((state,None))
    demos_all.append(d)
print "all demos", demos_all

#strip out actions for plotting the demos in figure
demos_states = [[sa[0] for sa in d] for d in demos_all]

plot_nav_world(size, features, demos_states)


