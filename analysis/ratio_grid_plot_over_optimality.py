import numpy as np
import matplotlib.pyplot as plt

#goal is to get 7 points from bad to good in terms of how well the evaluation
#policy matches the expert's policy
# I want to show the bound and accuracy as this changes to show that the method works over a wide variety of evaluation policies
#TODO Run a formal experiment using Q-learning to show this over a larger set


#I'm using results from ratio experiments 2,3,4,5, and 6

#experiment 2 has true 1.322864
#experiment 3 has 0.294115
#experiment 6 has 0.157834
#experiment 5 has 0.054425
#experiment 4 has 0.008376
#last data point should be 0

#actually I want to plot from 0 to 1.32 with bounds on y-axis

#I'm going to use alpha = 100 and chain_length = 2000 since they seem to level after that


mcmc_samples = [500, 1000, 2000, 5000]
alpha = 100
size = 5
num_reps = 100
tol = 0.0001

for num_mcmc_samples in mcmc_samples:
    accuracies = []
    average_bounds = []
    stdev_bounds = []
    true_perf_ratio = []
    for expmt in [4,5,6,3,2]:

        filename = "./grid_world/ratio_ctrpolicy" + str(expmt) + "_size"+str(size) +"_mcmc"+str(num_mcmc_samples)+"_reps" + str(num_reps) + "_alpha" + str(alpha) +  ".txt"           
        ratios = np.loadtxt(filename, delimiter='\t', skiprows=1)
        #print ratios
        #print np.nanmean(ratios,axis = 0)
        predicted = ratios[:,0]
        actual = ratios[:,1]
        accuracy = 0.0
        for i in range(len(predicted)):
            if predicted[i] >= actual[i] or np.abs(predicted[i] - actual[i]) < tol:
                accuracy += 1.0
        accuracy = accuracy / len(predicted)
        accuracies.append(accuracy)
        average_bounds.append(np.nanmean(predicted))
        true_perf_ratio.append(actual[0])
        stdev_bounds.append(2*np.nanstd(predicted))

    fig_cnt = 1
    plt.figure(fig_cnt)
    plt.title(r"5x5 navigation domain $\alpha=100$")
    plt.xlabel("true performance ratio")
    plt.ylabel("average upper bound")
    plt.errorbar(true_perf_ratio, average_bounds, yerr=stdev_bounds, fmt='o-', label="N = " + str(num_mcmc_samples))
    #plot dotted line across diagonal
    plt.plot([0,true_perf_ratio[-1]],[0,true_perf_ratio[-1]], 'k:')
    plt.legend(loc='best')

    #    plt.savefig("lower_bound_accuracy_95conf_mcmc_exp6.png")

    fig_cnt = 2
    plt.figure(fig_cnt)
    plt.title(r"5x5 navigation domain $\alpha=100$")
    plt.xlabel("true performance ratio")
    plt.ylabel("accuracy")
    plt.plot(true_perf_ratio, accuracies, 'o-',label="N = " + str(num_mcmc_samples))
    #plot 95% confidence line
    plt.plot([0,true_perf_ratio[-1]],[0.95, 0.95], 'k:')
    plt.legend(loc='best')
    #    plt.savefig("true_return_ratio_mcmc_exp6.png")
plt.figure(1)
plt.savefig("upperbound5x5alpha100_rangetest.eps")
plt.figure(2)
plt.savefig("accuracy5x5alpha100_rangetest.eps")
plt.show()
        
