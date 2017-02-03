from solvers import mdp
import copy
import numpy as np

def remove_duplicates(demos):
    demo_set = set()
    for d in demos:
        for sa in d:
            demo_set.add(sa)
    return [list(demo_set)]


def calculate_posterior(mdp, q, expert_demos, prior='uniform', alpha=1.0):
    z = []
    e = 1
    
    for s_e, a_e in expert_demos:
        #print s_e, a_e
        #print q[s_e,a_e]
        for a in mdp.actions(s_e):
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
    #print "eval mdp reward", eval_mdp.reward
    #print "evaluating at states", eval_mdp.init
    U = dict([(s, 0) for s in eval_mdp.states])
    U = mdp.policy_evaluation(pi, U, eval_mdp, k=100)
    #print "U", U
    return sum([p * U[s0] for (p, s0) in start_dist])    
    
#takes an estimated reward, demos, an MDP\R and evaluates the absolute difference of values if reward_eval is true reward
def policy_value_demo_diff(eval_pi, reward_sample, demos, mdp_model,
                            start_dist):
    
    #evaluate pi_est and demos on MDP\R + reward_eval
    mdp_eval = copy.deepcopy(mdp_model)
    mdp_eval.reward = reward_sample
    
    
    V_pi_est = evaluate_expected_return(eval_pi, mdp_eval, start_dist)
    V_demos = evaluate_expected_return_demos(demos, mdp_eval)
    #print "V_pi_est", V_pi_est
    #print "V_demos", V_demos
    #calculate the ratio
    abs_diff = np.abs(V_demos - V_pi_est)
    ###debug stuff
    #if abs_diff > 0.875:
    #    print "-----------------"
    #    print "V_pi_est", V_pi_est
    #    print "V_demos", V_demos
    #    mdp_eval.print_rewards()
    #    mdp_eval.print_arrows()
    #
    #    print "abs_diff", abs_diff
    
    return abs_diff
    
    
    #takes an estimated reward, demos, an MDP\R and evaluates the absolute difference of values if reward_eval is true reward
def policy_value_demo_ratio(eval_pi, reward_sample, demos, mdp_model,
                            start_dist):
    
    #evaluate pi_est and demos on MDP\R + reward_eval
    mdp_eval = copy.deepcopy(mdp_model)
    mdp_eval.reward = reward_sample
    
    
    V_pi_est = evaluate_expected_return(eval_pi, mdp_eval, start_dist)
    V_demos = evaluate_expected_return_demos(demos, mdp_eval)
    #print "V_pi_est", V_pi_est
    #print "V_demos", V_demos
    #calculate the ratio
    abs_ratio = np.abs(V_demos - V_pi_est) / np.abs(V_demos)
    ###debug stuff
#     if abs_ratio > 3:
#         print "-----------------"
#    print "V_pi_est", V_pi_est
#    print "V_demos", V_demos
#         #mdp_eval.print_rewards()
#         #mdp_eval.print_arrows()
#     
#    print "abs_ratio", abs_ratio
#     
    return abs_ratio


#computes the true difference between eval and true policies on true mdp
def policy_value_true_diff(eval_pi, true_pi, true_mdp, start_dist):
    #print eval_pi
    #print true_pi
    
    V_pi_est = evaluate_expected_return(eval_pi, true_mdp, start_dist)
    V_true = evaluate_expected_return(true_pi, true_mdp, start_dist)
    #print "V_pi_est", V_pi_est
    #print "V_demos", V_demos
    #calculate the ratio
    abs_diff = np.abs(V_true - V_pi_est)
    return abs_diff


#computes the true difference between eval and true policies on true mdp
def policy_value_true_ratio(eval_pi, true_pi, true_mdp, start_dist):
    #print eval_pi
    #print true_pi
    
    V_pi_est = evaluate_expected_return(eval_pi, true_mdp, start_dist)
    V_true = evaluate_expected_return(true_pi, true_mdp, start_dist)
    #print "V_pi_est", V_pi_est
    #print "V_true", V_true
    #calculate the ratio
    abs_ratio = np.abs(V_true - V_pi_est) / np.abs(V_true)
    return abs_ratio

    
def bound_return_diff_mcmc(eval_pi, opt_pi, true_mdp, demos, delta_conf, reward_samples, burn=0):
    reward_samples = reward_samples[burn:]
    num_samples = len(reward_samples)
    print "num samples", num_samples
    diffs = []
    #get initial starting distribution from mdp
    start_dist = [(1./len(true_mdp.init), s) for s in true_mdp.init]
    #print "start distribution", start_dist
    for r in reward_samples:
        
        #calculate the return difference between demos and eval policy
        diff = policy_value_demo_diff(eval_pi, r, demos,
                                       true_mdp, start_dist)
        #print "ratio=", ratio
        diffs.append(diff)
    #print "unsorted", diffs
    #sort the differences between demo return and map return and pick the (1-delta) percentile
    #print ratios
    ratios_ascending = np.sort(diffs)    
    ratios_descending = ratios_ascending[::-1]
    #print "sorted", ratios_descending
    upper_bnd_indx = np.floor((1-delta_conf) * num_samples)
    #print lower_bnd_indx
    print delta_conf, "th percentile", ratios_descending[upper_bnd_indx] 
    conf_diff = ratios_descending[upper_bnd_indx] 


    #compare the lower bound to the difference between the map return and the return of the optimal policy on the actual reward.
    #just use the true mdp
    true_diff = policy_value_true_diff(eval_pi, opt_pi, true_mdp,
                                         start_dist)
    print "true ratio = ", true_diff
    return conf_diff, true_diff, diffs

def bound_return_ratio_mcmc(eval_pi, opt_pi, true_mdp, demos, delta_conf, reward_samples, burn=0):
    reward_samples = reward_samples[burn:]
    num_samples = len(reward_samples)
    print "num samples", num_samples
    diffs = []
    #get initial starting distribution from mdp
    start_dist = [(1./len(true_mdp.init), s) for s in true_mdp.init]
    #print "start distribution", start_dist
    for r in reward_samples:
        
        #calculate the return difference between demos and eval policy
        diff = policy_value_demo_ratio(eval_pi, r, demos,
                                       true_mdp, start_dist)
        #print "ratio=", diff
        diffs.append(diff)
    #print "unsorted", diffs
    #sort the differences between demo return and map return and pick the (1-delta) percentile
    #print ratios
    ratios_ascending = np.sort(diffs)    
    ratios_descending = ratios_ascending[::-1]
    #print "sorted", ratios_descending
    upper_bnd_indx = int(np.floor((1-delta_conf) * num_samples))
    #print lower_bnd_indx
    print delta_conf, "th percentile", ratios_descending[upper_bnd_indx] 
    conf_diff = ratios_descending[upper_bnd_indx] 


    #compare the lower bound to the difference between the map return and the return of the optimal policy on the actual reward.
    #just use the true mdp
    true_diff = policy_value_true_ratio(eval_pi, opt_pi, true_mdp,
                                         start_dist)
    print "true ratio = ", true_diff
    return conf_diff, true_diff, ratios_descending

#use the bound from phil's paper on high confidence off-policy eval    
def chernoff_hoeffding(x, delta, b):
    return np.mean(x) - b * np.sqrt(np.log(1/delta)/(2*len(x)))
    

