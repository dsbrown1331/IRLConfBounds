import numpy as np
import matplotlib.pyplot as plt
from bounds import bound_methods
from numpy import nan, inf


#goal is to get 7 points from bad to good in terms of how well the evaluation
#policy matches the expert's policy
# I want to show the bound and accuracy as this changes to show that the method works over a wide variety of evaluation policies
#TODO Run a formal experiment using Q-learning to show this over a larger set


#These results are for using policies to compute expected value ratio


num_mcmc_samples = 2000
alpha = 100
size = 5
num_reps = 50
tol = 0.0001
burn = 0 #
delta_conf = 0.95
demos= [25]


for loss in [10]:
    data = []
    for num_demos in demos:
        samples_all = []
        for rep in range(num_reps):
            #print rep
            filename = "/home/dsbrown/workspace/IRLConfidenceBounds/paper_test/nav_world/grid5x5/gridnav_policy_ratio_size"+str(size) + "_qpolicyloss" + str(loss) + "_ndemos" + str(num_demos) + "_mcmc"+str(num_mcmc_samples)+"_rep" + str(rep) + "_alpha" + str(alpha) +  ".txt"
            #print filename
            f = open(filename,'r')   
            f.readline()                                #clear out comment from buffer
            actual = (float(f.readline())) #get the true ratio 
            #print "actual", actual
            f.readline()                                #clear out ---

            for line in f:                              #read in the mcmc chain
                val = float(line)                       #check if nan or inf and ignore
                if val != nan and val != inf:
                    samples_all.append(float(line))
            #print samples
            #burn 
        print "num demos", num_demos
        print "max", np.max(samples_all)
        print "std", np.std(samples_all)
        print "median", np.median(samples_all)
        
        data.append(samples_all)
    fig_cnt = 1
    plt.figure(fig_cnt)
    plt.title(str(num_demos) + " Demos " + r"5x5 navigation domain $\alpha = $" + str(alpha) + ", $N = $" + str(num_mcmc_samples))
    plt.xlabel("performance ratio", fontsize=18)
    plt.ylabel("frequency", fontsize=18)
    plt.hist(data,100, label=["D = " + str(num_demos) for num_demos in demos])
    #plt.plot(true_perf_ratio, average_bounds, fmts[bounds.index(bound_type)], label=bound_type, lw=1)
    #plot dotted line across diagonal
    plt.xticks(fontsize=15) 
    plt.yticks(fontsize=15) 
    plt.legend(loc='best', fontsize=18)
    
    plt.savefig("D" + str(num_demos) + "L" + str(loss) + ".png")
    
        

        #plt.savefig("policyaccuracy5x5alpha100_qtest_D" + str(num_demos) + ".eps")
    plt.show()
    
