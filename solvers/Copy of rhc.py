'''
Created on Feb 8, 2017

@author: dsbrown
'''
import mdp
import numpy as np
import timeit



state_horizon_values = {}

def generate_nonterminal_mdp(size, rewards):
    """
       returns an mdp without terminal state that is @size by @size large
       and has random rewards at each state drawn from the list of possible
       rewards given by @rewards
    """
    terminals =[] 
    init_dist = []
    true_reward = [[rewards[np.random.randint(len(rewards))] 
                    for _ in range(size)]
                    for _ in range(size)]
    true_mdp = mdp.GridMDP(true_reward,
                      terminals, init_dist, gamma=.95)
    return true_mdp

def generate_debug_mdp():
    """fixed mdp for debugging"""
    terminals =[(0,0)] 
    init_dist = []
    true_reward = [[0,0,0],
                   [0,-1,0],
                   [1,-1,0]]
    true_mdp = mdp.GridMDP(true_reward,
                      terminals, init_dist, gamma=.95)
    return true_mdp


def compute_qvals(state, horizon, mdp_model):
    """ for each state in the list of states compute the qvalue with horizon using alg in
        "Betweeen imitation and intention"
    """
    #recursion over all actions
    possible_actions = mdp_model.actions(state)
    Qsa = {}
    for a in possible_actions:
        next_states = [ps[1] for ps in mdp_model.T(state, a)] #get probability state tuples from T
        state_values = {}
        for s in next_states:
            state_values[s] = compute_value(s, horizon - 1, mdp_model)
        #compute q-values
        qval = mdp_model.R(state) + mdp_model.gamma * sum([p * state_values[s1] 
                                        for (p, s1) in mdp_model.T(state, a)])
        Qsa[(state, a)] = qval
    
    return Qsa
    
    
def compute_value(state, horizon, mdp_model):
    """recursive call to compute state value out to horizon"""
    #base case
    if horizon == 0:     
        return 0    
    
    #recursion over all actions
    possible_actions = mdp_model.actions(state)
    Qsah = []
    for a in possible_actions:
        next_states = [ps[1] for ps in mdp_model.T(state, a)] #get probability state tuples from T
        state_values = {}
        for s in next_states:
            if (s,horizon-1) not in state_horizon_values:
                state_horizon_values[(s,horizon-1)] = compute_value(s, horizon - 1, mdp_model)
        #compute q-values
        qval = mdp_model.R(state) + mdp_model.gamma * sum([p * state_horizon_values[(s1, horizon-1)] 
                                        for (p, s1) in mdp_model.T(state, a)])
        Qsah.append(qval)
    value_sh = np.max(Qsah)
    return value_sh         

        


def main():
    #parameters to run experiment
    #size = 3          #length of grid world

    #r_min = -1
    #r_max = 0
    #r_step = 1.0
    #rewards = np.arange(r_min,r_max+r_step,r_step)  #possible rewards to pick from
    
    #generate random MDP
    true_mdp = generate_nonterminal_mdp(20,[-1,0,1])
    true_reward = true_mdp.reward
    #print "true reward"
    #print true_reward

    #solve for optimal policy and q-values
    start = timeit.default_timer()
    expert_policy, true_U = mdp.policy_iteration(true_mdp)
    true_qvals = mdp.get_q_values(true_mdp, true_U)
    
    #Your statements here

    stop = timeit.default_timer()

    print "time to run DP", stop - start 

    print "q-values"
    print true_qvals
    
    
    #!!debug info
    #print "----debug----"
    #print "True rewards:"
    #true_mdp.print_rewards()
    #print "Expert policy:"
    #true_mdp.print_arrows()
    #!!end debug

    #print compute_value((2,0), 7, true_mdp)
    #print compute_qvals((2,0), 7, true_mdp)

    #print true_qvals
    #print "compare to"
    

    start = timeit.default_timer()
    qvals_rh = {}
    horizon = 100
    for state in list(true_mdp.states)[:100]:
        qvals_rh.update(compute_qvals(state, horizon, true_mdp))
    #print qvals_rh
    #Your statements here

    stop = timeit.default_timer()

    print "time to run RHC", stop - start 
    print qvals_rh == true_qvals
    print qvals_rh[((0,0),(0,1))]
    print true_qvals[((0,0),(0,1))]
    

if __name__=="__main__":
    main()