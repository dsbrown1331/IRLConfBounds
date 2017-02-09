import numpy as np
import matplotlib.pyplot as plt
from bounds import bound_methods
from numpy import nan, inf


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


mcmc_samples = [3000]
alpha = 100
size = 5
num_reps = 100
tol = 0.0001
burn = 0 #
delta_conf = 0.95
num_bootstrap = 1000
bounds = ["VAR", "MPeBC", "TT", "BS"]
fmts = ['o-','s--','^-.', '*:']
loss_ranges = {2:[5,7,9,15], 4:[5,7,9,11,12,14,15], 8:[3,5,7,9,11,12,14,15], 25:[3,5,6,7,8,9,10,11,12,13,14,15]}
for num_demos in [2,4,8,25]:
    for num_mcmc_samples in mcmc_samples:
        for bound_type in bounds:
            accuracies = []
            average_bounds = []
            stdev_bounds = []
            true_perf_ratio = []
            for loss in loss_ranges[num_demos]:
                predicted = []
                for rep in range(num_reps):
                    #print rep
                    filename = "/home/dsbrown/workspace/IRLConfidenceBounds/paper_test/nav_world/grid5x5/gridnav_ratio_size"+str(size) + "_qpolicyloss" + str(loss) + "_ndemos" + str(num_demos) + "_mcmc"+str(num_mcmc_samples)+"_rep" + str(rep) + "_alpha" + str(alpha) +  ".txt"
                    #print filename
                    f = open(filename,'r')   
                    f.readline()                                #clear out comment from buffer
                    actual = (float(f.readline())) #get the true ratio 
                    #print "actual", actual
                    f.readline()                                #clear out ---
                    samples = []
                    for line in f:                              #read in the mcmc chain
                        val = float(line)                       #check if nan or inf and ignore
                        if val != nan and val != inf:
                            samples.append(float(line))
                    #print samples
                    #burn 
                    burned_samples = samples[burn:]
                    #compute confidence bound
                    if bound_type == "VAR":
                        upper_bnd = bound_methods.value_at_risk(burned_samples, delta_conf)
                    elif bound_type == "MPeBC":
                        upper_bnd = bound_methods.phil_upper_bnd(burned_samples, delta_conf, 50)
                    elif bound_type == "TT":
                        upper_bnd = bound_methods.ttest_upper_bnd(burned_samples, delta_conf)
                    elif bound_type == "BS":                   
                        upper_bnd = bound_methods.bootstrap_percentile_confidence_upper(burned_samples, delta_conf, num_bootstrap)
                    
                    
                    #print "upper bound", upper_bnd
                    predicted.append(upper_bnd)
                accuracy = 0.0
                for i in range(len(predicted)):
                    if (predicted[i] >= actual) or np.abs(predicted[i] - actual) < tol:
                        accuracy += 1.0
                accuracy = accuracy / len(predicted)
                print bound_type
                print "loss", loss, "accuracy", accuracy
                accuracies.append(accuracy)
                average_bounds.append(np.nanmean(predicted))
                true_perf_ratio.append(actual)
                print "true bound", actual
                print "predicted bounds", predicted
                stdev_bounds.append(np.nanstd(predicted))
                print 
        
            fig_cnt = 1
            plt.figure(fig_cnt)
            plt.title(str(num_demos) + " Demos " + r"5x5 navigation domain $\alpha=100$, $N = 3000$")
            plt.xlabel("true performance ratio", fontsize=18)
            plt.ylabel("average upper bound", fontsize=18)
            #plt.errorbar(true_perf_ratio, average_bounds, yerr=stdev_bounds, fmt=fmts[bounds.index(bound_type)], label=bound_type, lw=1)
            plt.plot(true_perf_ratio, average_bounds, fmts[bounds.index(bound_type)], label=bound_type, lw=1)
            #plot dotted line across diagonal
            plt.plot([0,true_perf_ratio[-1]],[0,true_perf_ratio[-1]], 'k:')
            plt.xticks(fontsize=15) 
            plt.yticks(fontsize=15) 
            plt.legend(loc='best', fontsize=18)
        
            #    plt.savefig("lower_bound_accuracy_95conf_mcmc_exp6.png")
        
            fig_cnt = 2
            plt.figure(fig_cnt)
            plt.title(str(num_demos) + " Demos " + r"5x5 navigation domain, $\alpha=100$, $N = 3000$")
            plt.xlabel("true performance ratio", fontsize=17)
            plt.ylabel("accuracy", fontsize=17)
            plt.plot(true_perf_ratio, accuracies, 'o-',label= bound_type, lw = 1)
            #plot 95% confidence line
            plt.plot([0,true_perf_ratio[-1]],[0.95, 0.95], 'k:')
            plt.xticks(fontsize=15) 
            plt.yticks(fontsize=15)
            plt.legend(loc='best',fontsize=18)
            #    plt.savefig("true_return_ratio_mcmc_exp6.png")
    plt.figure(1)
    plt.savefig("upperbound5x5alpha100_qtest_D" + str(num_demos) + ".eps")
    plt.figure(2)
    plt.savefig("accuracy5x5alpha100_qtest_D" + str(num_demos) + ".eps")
    plt.show()
        
