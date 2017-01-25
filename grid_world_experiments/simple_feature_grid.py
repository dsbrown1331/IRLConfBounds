import solvers
import copy
import numpy as np
from scipy.misc import logsumexp
import birl.feature_birl
#test of whether the bootstrap from BIRL posterior gives us an estimate of performance that 
#upper bounds with certain confidence level
#Version 5.0
#fix the terminal reward to be zero and use Rmin = -5 and Rmax= -1 to better compare with 
#the random experiment.
#also remove duplicates from demos (s,a) pairs
#also burn more
#also skip samples
#keep number of samples the same and increase size of world (6x6)
##don't skip but just run really long chains to see if skipping is really all that good.

def remove_duplicates(demos):
    demo_set = set()
    for d in demos:
        for sa in d:
            demo_set.add(sa)
    return [list(demo_set)]


def calculate_posterior(solvers, q, expert_demos, prior='uniform', alpha=1.0):
    z = []
    e = 1
    
    for s_e, a_e in expert_demos:
        #print s_e, a_e
        #print q[s_e,a_e]
        for a in solvers.actions(s_e):
            #print s_e, a
            #print q[s_e, a]
            z.append(np.exp(alpha * q[s_e, a])) #normalizing constant in denominator
        #print "prob", np.exp(alpha * q[s_e, a_e]) / np.sum(z)
        e *=  np.exp(alpha * q[s_e, a_e]) / np.sum(z)#log(e^(alpha * Q) / sum e^Q)
        #print e
        
        del z[:]  #Removes contents of Z
    #TODO get a better prior and maybe use state info, not just raw values??
    if prior is 'uniform':
        return e #priors will cancel in ratio #TODO figure out how to do uniform?
    # return P(demo | R) * P(R) in log space

def merge_trajectories(demos):
    merged = []  
    for demo in demos:
        merged.extend(demo)  
    return merged

def evaluate_expected_return_demos(demo_set, eval_mdp):
    #print "evaluating demos"
    expected_reward = 0.0
    for demo in demo_set:
        t = 0
        for s,a in demo:
            #print "reward", eval_mdp.R(s), " for state", s
            expected_reward += (eval_mdp.gamma ** t)  * eval_mdp.R(s)
            #print expected_reward
            t += 1
    return expected_reward / len(demo_set)

#one that works for non-optimal policy (different than one induced by reward
def evaluate_expected_return(pi, eval_mdp, start_dist):
    #print "eval solvers reward", eval_mdp.reward
    #print "evaluating at states", eval_mdp.init
    U = dict([(s, 0) for s in eval_mdp.states])
    U = solvers.mdp.policy_evaluation(pi, U, eval_mdp, k=100)
    #print "U", U
    return sum([p * U[s0] for (p, s0) in start_dist])    
    
#takes an estimated reward, demos, an MDP\R and evaluates the absolute difference of values if reward_eval is true reward
def policy_value_demo_diff(learned_mdp, reward_eval, demos, start_dist):
    #print "reward_est", reward_est
    #print "reward_eval", reward_eval
    #first get optimal policy if reward_est is true
    pi_est, U_est = solvers.mdp.policy_iteration(learned_mdp)
    #print "pi_est", pi_est
    #evaluate pi_est and demos on MDP\R + reward_eval
    mdp_eval = copy.deepcopy(learned_mdp)
    mdp_eval.reward = reward_eval
    V_pi_est = evaluate_expected_return(pi_est, mdp_eval, start_dist)
    V_demos = evaluate_expected_return_demos(demos, mdp_eval)
    #print "V_pi_est", V_pi_est
    #print "V_demos", V_demos
    #calculate the ratio
    abs_diff = np.abs(V_demos - V_pi_est)
    return abs_diff


def estimate_return_lower_bound(true_mdp, num_demos, delta_conf, mcmc_step_size, chain_length, burn, features):
    
    #compute the optimal policy for the actual reward
    opt_U = solvers.mdp.value_iteration(true_mdp)
    opt_qvals = solvers.mdp.get_q_values(true_mdp, opt_U)
    #true_mdp.print_arrows()

    #generate X demonstrations
    demo_states = [s for s in true_mdp.states if s not in true_mdp.terminals]
    #print demo_states
    #print demo_states
    opt_demos = []
    rand_start_idxs = np.random.choice(len(demo_states),num_demos, replace=False)
    #rand_start_idxs = [2]
    #print [demo_states[idx] for idx in rand_start_idxs]
    #print rand_start_idxs
    s0_dist = []
    for i in range(len(rand_start_idxs)): 
        #pick random initial state
        rand_init = demo_states[rand_start_idxs[i]]
        s0_dist.append(rand_init)
        #print rand_init
        demo = solvers.mdp.monte_carlo_argmax_rollout(true_mdp, opt_qvals, rand_init, rollout_length)
        opt_demos.append(demo)
    

    #train BIRL on the first X-1 demos to get MAP reward and posterior
    #TODO do n-fold cross validation
    train_demos = remove_duplicates(opt_demos[:])
    #print "demos",train_demos
    #print "train_demos", train_demos
    #test_demos = opt_demos[num_train:]
    test_demos = opt_demos[:]
    #print "using train as test but with duplicates"
    #print "test_demos", test_demos
    print "training on ", num_demos, "demos"
    #print opt_demos
    #print features
           
    birl = birl.feature_birl.BIRL_FEATURE(train_demos,            
                    true_mdp.get_grid_size(), true_mdp.terminals, 
                    true_mdp.init, features,
                     birl_iteration=chain_length,
                     r_min=r_min,r_max=r_max,
                     step_size=mcmc_step_size)

    chain, map_mdp =  birl.run_birl()
    #calculate the mean reward and policy since there is a theorem for it
    mean_reward = birl.feature_birl.average_chain(chain, burn)
    mean_mdp = copy.deepcopy(map_mdp)
    mean_mdp.reward = mean_reward
    
    #print chain
    print "Mean estimate"
    mean_mdp.print_rewards()
    mean_mdp.print_arrows()
    #print "weights", mean_mdp.weights #currently doesn't recover the weights
    
    #compute the Mean policy
    map_U = solvers.mdp.value_iteration(mean_mdp)
    map_qvals = solvers.mdp.get_q_values(mean_mdp, map_U)
    
    #test out likelihoods
    #print "birl"
    #print calculate_posterior(map_mdp, map_qvals, merge_trajectories(train_demos), prior='uniform', alpha=1.0)
    #print "opt"
    #print calculate_posterior(true_mdp, opt_qvals, merge_trajectories(train_demos), prior='uniform', alpha=1.0)

    #sample B rewards from posterior
    posterior = chain[burn:]
    print len(posterior)
    print "number of samples", len(posterior)
    print "using full posterior P(R|D) as bootstrap sample"
    #sample_idxs = np.random.choice(len(posterior),num_bootstrap, replace=True)
    #print sample_idxs
    #bootstrap_sample = [posterior[idx] for idx in sample_idxs]
    bootstrap_sample = posterior[:]
    print "number of samples", len(bootstrap_sample)
    num_bootstrap = len(bootstrap_sample)
    
    #print bootstrap_sample
    #for r in bootstrap_sample:
    #    test_mdp = copy.deepcopy(true_mdp)
    #    test_mdp.reward = copy.deepcopy(r)
    #    test_mdp.print_rewards()
    #    test_mdp.print_arrows()
    

    #for each reward calculate the return of the test demos on each b in B and calculate the return of the MAP policy starting at the same state as the demo
    #calc start_distribution for test initial states
    diffs = []
    non_terminal_states = [s for s in true_mdp.states if s not in true_mdp.terminals]
    start_dist = [(1./len(s0_dist), s) for s in s0_dist]
    print "start distribution", start_dist
    for r in bootstrap_sample:
        
        #calculate the return ratio between demos and learned policy
        #TODO change train_demos to test_demos 
        diff = policy_value_demo_diff(map_mdp, r, test_demos, start_dist)
        #print "ratio=", ratio
        diffs.append(diff)
    
    #sort the differences between demo return and map return and pick the (1-delta) percentile
    #print ratios
    ratios_ascending = np.sort(diffs)    
    ratios_descending = ratios_ascending[::-1]
    #print ratios_descending
    lower_bnd_indx = np.floor((1-delta_conf) * num_bootstrap)
    #print lower_bnd_indx
    print delta_conf, "th percentile", ratios_descending[lower_bnd_indx] 
    conf_diff = ratios_descending[lower_bnd_indx] 


    #compare the lower bound to the difference between the map return and the return of the optimal policy on the actual reward.
    #just use the true solvers
    true_diff = policy_value_demo_diff(map_mdp, true_mdp.reward, test_demos, start_dist)
    print "true ratio = ", true_diff
    return conf_diff, true_diff


if __name__=="__main__":

    num_reps = 100
    burn = 0
    size = 4
    weights = [0.5,0,-0.5]
    grid = [[0 for _ in range(size)] for _ in range(size)]
    
    for mcmc_length in [1000]:
        for num_demos in range(1,size*size):
            rows = size
            cols = size
            rollout_length = rows*cols - 1 #ends in terminal so shouldn't matter
            delta_conf = 0.95
            terminals=[(0,0)]
            init = []
            gamma = 0.99
            r_min = -1.0
            r_max = 1.0
            mcmc_step_size = 0.1
            chain_length = mcmc_length            
            
            
            f = open("safety_test/features/three_feature_mean_mcmc1_"+str(rows) + "x" +str(cols) +"mcmc"+str(chain_length)+"num_demos"+str(num_demos)+"rmin"+str(r_min)+"rmax"+str(r_max)+"step"+str(mcmc_step_size) +".txt",'w')
            
            f.write("predicted\tactual\n")
            for rep in range(num_reps):
                print "--------------rep",rep,"--------------"
                #make random features
                features = {}
                features[(0,0)] = [1,0,0]
                for x in range(rows):
                    for y in range(cols):
                        if (x,y) == (0,0):
                            pass
                        else:
                            f_rand = [0,np.random.rand(),np.random.rand()]
                            features[(x,y)] = f_rand/np.linalg.norm(f_rand,1) #normalize

                true_mdp = solvers.mdp.DeterministicWeightGridMDP(features,
                      weights, grid, terminals=terminals, init=init,
                       gamma=gamma,r_min=r_min,r_max=r_max)
                
                true_mdp.print_rewards()
                

                conf_diff, true_diff = estimate_return_lower_bound(
                true_mdp, num_demos, delta_conf,
                     mcmc_step_size, chain_length, burn, features)
                #write to file
                f.write("%f\t%f\n" % (conf_diff, true_diff))
            f.close()

