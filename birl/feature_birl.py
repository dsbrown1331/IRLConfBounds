"""
Daniel Brown
based on aima-based-irl
"""

from solvers.mdp import *
from solvers.utils import *
from copy import deepcopy
from math import exp


#takes 
class BIRL_FEATURE():
    def __init__(self, expert_trace, grid_size, terminals, init, features, gamma, step_size=1.0, r_min=-10.0,
                 r_max=10.0, prior = 'uniform', birl_iteration = 2000, prior_dict = {}, alpha=1.0):
        self.n_rows, self.n_columns = grid_size
        self.r_min, self.r_max = r_min, r_max
        self.step_size = step_size
        self.gamma = gamma
        #print "step size", self.step_size
        #print "r_min", self.r_min
        #print "r_max", self.r_max
        self.expert_trace = merge_trajectories(expert_trace)
        self.features = features
        self.num_features = len(features[features.keys()[0]])
        #print 'trace for batch', self.expert_trace
        self.terminals = terminals
        self.init = init
        self.alpha = alpha
        self.prior = prior #string to specify what type of prior
        #print 'prior', self.prior
        self.birl_iteration = birl_iteration #how long to run the markov chain
        self.prior_dict = prior_dict
       
    def run_birl(self):
        #This is the core BIRL algorithm
        Rchain = [] #store rewards along the way, #TODO dictionaries are probably not best...
        #TODO make this a random reward vector
        mdp = self.create_zero_weights() #pick a starting reward vector
        #Rchain.append(solvers.reward) #I don't think i want the initital random reward
        
        #print 'old rewards'
        #solvers.print_rewards()
        pi, u = policy_iteration(mdp) #calculate optimal policy and utility for random R
        q = get_q_values(mdp, u) #get the q-values for R in solvers
        #print "qqqqqqq"
        #print q
        #print "qqqqqq"
        posterior = calculate_posterior(mdp, q, self.expert_trace, self.prior, self.prior_dict, self.alpha)
        bestPosterior = posterior 
        bestMDP = mdp
        for i in range(self.birl_iteration):
            if i % 10 == 0:
                print "===== iter", i, "======"
            #print i
            #print self.birl_iteration/10 
            #if i%(self.birl_iteration/10) == 0:
            #    print '.'
            new_mdp = deepcopy(mdp)
            new_mdp.modify_rewards_randomly(step = self.step_size) #pick random reward along grid
            #print 'new rewards'
            #new_mdp.print_rewards()
            #TODO this isn't exactly like the paper...
            #TODO ask scott about Q, should it be based on pi or pi^*
            new_u = policy_evaluation(pi, u, new_mdp) #evaluate old policy with old u and update using new R #changed to use default k which i changed to be 100
            #i wonder if it helps to start with old u? I guess a lot won't change 
            #check if there is a state where new action is better than old policy
            if pi != best_policy(new_mdp, new_u):
            #also try 
            #if q != get_q_values(new_mdp, new_u): #I think it is the same...
                #print 'old policy not optimal under new reward'
                new_pi, new_u = policy_iteration_warm_start(new_mdp,20,pi,u) #get new policy #TODO i think we could use the best_policy command above to speed things up
                new_q = get_q_values(new_mdp, new_u) #get new q-vals to calc posterior
                new_posterior = calculate_posterior(new_mdp, new_q, self.expert_trace, self.prior, self.prior_dict, self.alpha)
                ##print 'prob switch', exp(new_posterior - posterior)
                #print new_posterior
                #print posterior
                if (new_posterior - posterior) > 1 or probability(min(1, exp(new_posterior - posterior))):
                    #I figure out why it doesn't switch very often, there was a bug in policy evaluation
                    #TODO I could do much better I think using simulated annealing or randomized hill climbing...just systemtatically try neighboring rewards and climb with some noise...isn't that what mcmc does, though, I don't know if it's all that efficient...maybe a better prior is needed
                    #TODO why does switching happen that results in worse policy is it just the weighting by Qvals?
                    #print "===== iter", i, "======" 
                    ##print 'switched better'
                    ##print 'new rewards'
                    ##new_mdp.print_rewards()
                    #new_mdp.print_arrows()
                    #try saving the best so far
                    if bestPosterior < new_posterior:
                        bestPosterior = new_posterior
                        bestMDP = new_mdp
                        #print "best", i
                        #bestMDP.print_rewards()
                        #bestMDP.print_arrows()
                    
                    pi, u, mdp, posterior = new_pi, new_u, deepcopy(new_mdp), new_posterior

            else:
                new_q = get_q_values(new_mdp, new_u)
                new_posterior = calculate_posterior(new_mdp, new_q, self.expert_trace, self.prior, self.prior_dict, self.alpha)

                if (new_posterior - posterior) > 1 or probability(min(1, exp(new_posterior - posterior))):
                    #print "===== iter", i, "======" 
                    ##print 'switched random'
                    ##print 'new rewards'
                    #new_mdp.print_rewards()
                    #new_mdp.print_arrows()
                    mdp, posterior = deepcopy(new_mdp), new_posterior

            #print "iter", i
            #solvers.print_rewards()
            #print "---"

            Rchain.append(mdp.reward)
        return Rchain, bestMDP
        
        #------------- Reward functions ------------

    def create_zero_weights(self):
        weights = [0 for _ in range(self.num_features)]
        grid = [[0 for _ in range(self.n_columns)] for _ in range(self.n_rows)]
        return DeterministicWeightGridMDP(self.features, weights, grid, terminals=deepcopy(self.terminals), init = deepcopy(self.init), r_min = self.r_min, r_max = self.r_max, gamma = self.gamma)
                       
                       
    def create_rand_rewards(self):
        rand_r = [[np.random.randint(self.r_min, self.r_max+1) 
                        for c in range(self.n_columns)] 
                        for r in range(self.n_rows)]
        if self.prior == 'state-specific':
            for state in self.prior_dict:
                print state
                rand_r[self.n_rows-1 - state[1]][state[0]] = self.prior_dict[state]
        return GridMDP(rand_r, 
                        terminals=deepcopy(self.terminals), 
                        init = deepcopy(self.init), r_min=self.r_min,
                        r_max=self.r_max, gamma=self.gamma)

#seems correct, gives the log (P(demo | R) * P(R))
#TODO what do you do for the terminal state? when actions are None what do you normalize by
#TODO does the agent even know the terminals? what is the action in the terminal state?
#TODO anneal the alpha ?
#TODO I think I need to reverse the order of the iteration need to think about it more...
#overriden method
def calculate_posterior(mdp, q, expert_demos, prior, prior_dict, alpha):
    z = []
    e = 0
    
    for s_e, a_e in expert_demos:
        #print s_e, a_ed
        for a in mdp.actions(s_e):
            #print q[s_e, a]
            #print alpha
            z.append(alpha * q[s_e, a]) #normalizing constant in denominator
        #print q[s_e,a_e]
        e += alpha * q[s_e, a_e] - logsumexp(z) #log(e^(alpha * Q) / sum e^Q)
        #print e
        
        del z[:]  #Removes contents of Z
    #TODO get a better prior and maybe use state info, not just raw values??
    if prior == 'uniform':
        return e #priors will cancel in ratio #TODO figure out how to do uniform?
    elif prior == 'state-specific': #check if violates prior_dict for state specific rewards
        for state in prior_dict:
            if prior_dict[state] != mdp.R(state):
                return -10000 #log(0)
        return e 
        
    # return P(demo | R) * P(R) in log space


    
#take a bunch of lists of (s,a) tuples merge all into one list
def merge_trajectories(demos):
    merged = []  
    for demo in demos:
        merged.extend(demo)  
    return merged
#demos = [[((2, 1), (0, -1)), ((2, 0), (-1, 0)), ((1, 0), (-1, 0)), ((0, 0), (0, 1)), ((0, 1), None)], [((1, 1), (-1, 0)), ((0, 1), None)]]

#print merge_trajectories(demos)

#compute the average reward over the chain of rewards
def average_chain(chain, burn):
    mean_reward = {}
    count = 1.0
    #initialize
    for item in chain[burn]:
        mean_reward[item] = chain[burn][item]

    #add up all rewards
    for i in range(burn+1,len(chain)):
        count += 1.0
        for item in chain[i]:
            mean_reward[item] += chain[i][item]

    #calculate average
    for thing in mean_reward:
        mean_reward[thing] = mean_reward[thing] / count
    return mean_reward    
