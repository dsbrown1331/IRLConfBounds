'''
Created on Feb 8, 2017

@author: dsbrown
'''
import mdp
import numpy as np

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

        

