from solvers import mdp
from birl import feature_birl
from birl import birl_util
from bounds import confidence_bounds
import random
import copy
import cPickle as pickle
from domains.navgrid_test_plot import *
#Testing for a fixed world and fixed random policy how the conf bounds compare to real performance gap
#sticking with alpha = 100
#testing results for larger chains on toy problem
#testing more reps
#larger burn

##Testing optimal policy, actually, I can prove it will always be zero!

##BIRL PARAMS
mcmc_step = 0.1
burn = 0 #don't burn until afterwards
chain_lengths = [300]
test_alphas = [100]   #seems to really affect how close we match the expert! with low values we get differences. with 100+ we get exact match to demos even in only a couple iterations like 20 steps of mcmc

##EXPERIMENT PARAMS
num_reps = 1
delta_conf = 0.95

##MDP PARAMS
r_min = -1
r_max = 1
size = 9
grid = [[0 for _ in range(size)] for _ in range(size)]
gamma = 0.95

####make random world
init_states = [(0,0),(0,8),(8,0),(8,8),(5,0)]
term_states = [(4,4)]
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
# features = {}
# for row in range(size):
#     for col in range(size):
#         #pick random color for non_terminals
#         rand_color = random.choice(non_term_colors)
#         features[(col,row)] = rand_color
# #add terminal state
# for term in term_states:
#     features[term] = blue_f
# pickle.dump(features, open( "../grid_nav_world9x9.p", "wb" ) )
features = pickle.load( open( "../grid_nav_world9x9.p", "rb" ) )
features[(1,5)] = white_f
print features


true_mdp = mdp.DeterministicWeightGridMDP(features,
                weights, grid, terminals=term_states, init=init_states,
                gamma=gamma,r_min=r_min,r_max=r_max)
true_mdp.init = true_mdp.states
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


#use bad policy, always go right, as evaluation policy
#rand_pi = {s:(1,0) for s in true_mdp.states}
#rand_pi = {s:random.choice(true_mdp.actlist) for s in true_mdp.states}
rand_pi = copy.deepcopy(opt_pi)
#tweak two states, one is equally good, the other is suboptimal but not dangerous

rand_pi[(0,1)] = (1,0)
#rand_pi[(4,4)] = (-1,0)
#print "random policy", rand_pi
print rand_pi == opt_pi



for alpha in test_alphas:
    print "alpha",alpha
    for num_mcmc_samples in chain_lengths:
        print "chain length", num_mcmc_samples
        f = open("../paper_test/nav_world/gridnav_size"+str(size) +"_mcmc"+str(num_mcmc_samples)+"_reps" + str(num_reps) + "_alpha" + str(alpha) +  ".txt",'w')
        
        #f.write("predicted\tactual\n")
        for rep in range(num_reps):
            print rep
            print "using ", num_mcmc_samples, " samples"
            demos = demos_all
            ###calculate the evaluation policy, keep this fixed for now
            ##using the Mean BIRL since it has a theorem     
            birl = feature_birl.BIRL_FEATURE(demos,            
                                true_mdp.get_grid_size(), true_mdp.terminals, 
                                true_mdp.init, true_mdp.features, gamma,
                                 birl_iteration=num_mcmc_samples,
                                 r_min=r_min,r_max=r_max,
                                 step_size=mcmc_step, alpha=alpha)

            #TODO keep track of policies in BIRL to speed things up
            chain, map_mdp =  birl.run_birl()


            #print "MAP MDP"
            #map_mdp.print_rewards()
            #map_mdp.print_arrows()


            #print "Mean MDP"
            #mean_mdp.print_rewards()
            #mean_mdp.print_arrows()
                
            #TODO make sure that they are wrt to values not whatever action is argmax
            ###calculate performance bounds rep times to get accuracy
            conf_diff, true_diff, ratios = confidence_bounds.bound_return_ratio_mcmc(rand_pi, opt_pi, true_mdp, demos, delta_conf, chain)
            print ratios
            f.write("%f\n" % (true_diff))
            f.write("---\n")
            for r in ratios:
                f.write(str(r) + "\n")
        f.close()

