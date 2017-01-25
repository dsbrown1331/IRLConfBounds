import mdp
import my_birl_feature
import birl_util
import confidence_bounds
import random
#Testing for a fixed world and fixed random policy how the conf bounds compare to real performance gap


##MDP PARAMS
r_min = -1
r_max = 1
size = 5
grid = [[0 for _ in range(size)] for _ in range(size)]
gamma = 0.9

##BIRL PARAMS
mcmc_step = 0.1
burn = 0
chain_lengths = [10,50, 100, 200, 500, 1000, 2000]
test_alphas = [1,5,50,100,500]   #seems to really affect how close we match the expert! with low values we get differences. with 100+ we get exact match to demos even in only a couple iterations like 20 steps of mcmc

##EXPERIMENT PARAMS
num_reps = 25
delta_conf = 0.95

####make world
init_states = []
term_states = [(2,2)]
weights = [0,-1,1,0,0]
#Features are White, Red, Blue, Yellow, Green
white_f = [1,0,0,0,0]
red_f = [0,1,0,0,0]
blue_f = [0,0,1,0,0]
yellow_f = [0,0,0,1,0]
green_f = [0,0,0,0,1]
features = {(0,0) : white_f,
            (1,0) : white_f,
            (2,0) : red_f,
            (3,0) : yellow_f,
            (4,0) : white_f,
            (0,1) : yellow_f,
            (1,1) : green_f,
            (2,1) : white_f,
            (3,1) : white_f,
            (4,1) : white_f,
            (0,2) : white_f,
            (1,2) : white_f,
            (2,2) : blue_f,
            (3,2) : white_f,
            (4,2) : green_f,
            (0,3) : white_f,
            (1,3) : red_f,
            (2,3) : white_f,
            (3,3) : white_f,
            (4,3) : white_f,
            (0,4) : white_f,
            (1,4) : green_f,
            (2,4) : yellow_f,
            (3,4) : red_f,
            (4,4) : white_f}


true_mdp = mdp.DeterministicWeightGridMDP(features,
                weights, grid, terminals=term_states, init=init_states,
                gamma=gamma,r_min=r_min,r_max=r_max)
true_mdp.init = true_mdp.states
true_mdp.print_rewards()
true_mdp.print_arrows()

### use hard coded demos from optimal policy
opt_pi, opt_U = mdp.policy_iteration(true_mdp)
demos_all = []
for s0 in true_mdp.init:
    d = []
    state = s0
    while state not in true_mdp.terminals:
       action = opt_pi[state]
       d.append((state, action))
       state = true_mdp.go(state,action)
    d.append((state,None))
    demos_all.append(d)
print demos_all

#use bad policy, always go right, as evaluation policy
rand_pi = {s:(1,0) for s in true_mdp.states}
#print "random policy", rand_pi

for alpha in test_alphas:
    print "alpha",alpha
    for num_mcmc_samples in chain_lengths:
        print "chain length", num_mcmc_samples
        f = open("paper_test/grid_world/ctrpolicy_size"+str(size) +"_mcmc"+str(num_mcmc_samples)+"_reps" + str(num_reps) + "_alpha" + str(alpha) +  ".txt",'w')
        
        f.write("predicted\tactual\n")
        for rep in range(num_reps):
        
            print "using ", num_mcmc_samples, " samples"
            demos = demos_all
            ###calculate the evaluation policy, keep this fixed for now
            ##using the Mean BIRL since it has a theorem     
            birl = my_birl_feature.BIRL_FEATURE(demos,            
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
            conf_diff, true_diff = confidence_bounds.bound_return_diff_mcmc(rand_pi, opt_pi, true_mdp, demos, delta_conf, chain)
            
            f.write("%f\t%f\n" % (conf_diff, true_diff))
        f.close()

