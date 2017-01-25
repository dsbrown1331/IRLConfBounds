import mdp
import copy

def calculate_mean_mdp(chain, burn, mdp_model):
              
    #calculate the mean reward and policy since there is a theorem for it
    mean_reward = average_chain(chain, burn)
    mean_mdp = copy.deepcopy(mdp_model)
    mean_mdp.reward = mean_reward
    return mean_mdp

    

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
