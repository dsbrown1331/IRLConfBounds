# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:17:18 2017
Empirical Bootstrap method
@author: daniel
"""

#given a list of elements, compute mean and sample with replacement multiple
#times to get a bunch of sample deviations from mean

import numpy as np
import scipy





def bootstrap_confidence(sample_data, delta_conf, num_bootstrap):
    """Returns a 1-delta confidence bound on range of mean
       delta_confidence is actual confidence not 1-, e.g. delta_conf = 95%
    """
    
    sample_mean = np.mean(sample_data)
    #print "sample mean", sample_mean
    num_to_sample = len(sample_data)
    bootstrap_means = np.array([np.mean(np.random.choice(sample_data, num_to_sample, 
                                replace=True)) for i in range(num_bootstrap)])
    bootstrap_diffs = bootstrap_means - sample_mean
    sorted_diffs = np.sort(bootstrap_diffs)
    #print sorted_diffs
    #find index of 2.5 percentile and 97.5 percentile
    lower_bnd_indx = max(int(np.floor((1 - delta_conf) / 2.0 * num_bootstrap)) - 1,0)
    #print "lower index", lower_bnd_indx
    #print "lower bound diff", sorted_diffs[lower_bnd_indx]
    upper_bnd_indx = int(np.ceil((delta_conf + (1 - delta_conf) / 2.0) * num_bootstrap)) - 1
    #print "upper index", upper_bnd_indx
    #print "upper bound diff", sorted_diffs[upper_bnd_indx]
    lower_bnd = sample_mean - sorted_diffs[upper_bnd_indx]
    upper_bnd = sample_mean - sorted_diffs[lower_bnd_indx]
    #print "95\% conf bound", (lower_bnd, upper_bnd)
    return lower_bnd, upper_bnd
####!!!!!!!!!!!!!!! I think there is a bug in my code where I should be subtracting one from bounds in conf_bounds.py

def bootstrap_confidence_upper(sample_data, delta_conf, num_bootstrap):
    """Returns a 1-delta confidence bound on range of mean"""
    sample_mean = np.mean(sample_data)
    #print "sample mean", sample_mean
    num_to_sample = len(sample_data)
    bootstrap_means = np.array([np.mean(np.random.choice(sample_data, num_to_sample, 
                                replace=True)) for i in range(num_bootstrap)])
    bootstrap_diffs = bootstrap_means - sample_mean
    sorted_diffs = np.sort(bootstrap_diffs)
    #print sorted_diffs
    #find index of 2.5 percentile and 97.5 percentile
    lower_bnd_indx = max(int(np.floor((1 - delta_conf) * num_bootstrap)) - 1,0)
    #print "lower index", lower_bnd_indx
    #print "lower bound diff", sorted_diffs[lower_bnd_indx]
    upper_bnd = sample_mean - sorted_diffs[lower_bnd_indx]
    #print "95\% conf bound", upper_bnd
    return upper_bnd
    
def value_at_risk(sample_data, delta_conf):
    data_descending = np.sort(sample_data)[::-1]
    #print "sorted", ratios_descending
    upper_bnd_indx = int(np.floor((1-delta_conf) * len(sample_data)))
    return data_descending[upper_bnd_indx] 

def phil_lower_bnd(sample_data, delta_conf, c):
    N = float(len(sample_data))
    #print "N", N
    delta = 1 - delta_conf
    #print "delta", delta
    #trucate data
    #print "sample data", sample_data
    sample_data_trunc = np.array([min(s,c) for s in sample_data])
    #print "truncated data", sample_data_trunc
    #print sample_data_trunc / c
    #print (sample_data_trunc / c )  ** 2
    lower_bnd = (c / N) * (np.sum(sample_data_trunc / c) 
            - (7.0*N*np.log(2.0/delta))/(3.0*(N-1.0)) 
            - np.sqrt( ((2.0*np.log(2.0/delta))/(N-1.0)) 
            * (N * np.sum( (sample_data_trunc / c) ** 2) 
            - (np.sum(sample_data_trunc / c)) ** 2) ))
    return lower_bnd
    
def phil_upper_bnd(sample_data, delta_conf, c):
    N = float(len(sample_data))
    #print "N", N
    delta = 1 - delta_conf
    #print "delta", delta
    #trucate data
    #print "sample data", sample_data
    sample_data_trunc = np.array([min(s,c) for s in sample_data])
    #print "truncated data", sample_data_trunc
    #print sample_data_trunc / c
    #print (sample_data_trunc / c )  ** 2
    upper_bnd = (c / N) * (np.sum(sample_data_trunc / c) 
            + (7.0*N*np.log(2.0/delta))/(3.0*(N-1.0)) 
            + np.sqrt( ((2.0*np.log(2.0/delta))/(N-1.0)) 
            * (N * np.sum( (sample_data_trunc / c) ** 2) 
            - (np.sum(sample_data_trunc / c)) ** 2) ))
    return upper_bnd
    
def ttest_upper_bnd(sample_data, delta_conf):
    #compute mean and sample stdev
    m = len(sample_data)
    sample_mean = np.mean(sample_data)
    sample_std =  np.std(sample_data, ddof=1)
    t_val = scipy.stats.t.ppf(delta_conf, m - 1)
    return sample_mean + sample_std / np.sqrt(m) * t_val
    

#testing scripts for bounds
def main():
    sample_data = np.array([1,2,3,2,1,2,1,2,1,1,1,1,1,100,100,1,1,2,4,5,6,7])
    #sample_data = np.array(np.random.rand(3000))
    num_b = 10000
    delta_conf = 0.95
    print "sample mean", np.mean(sample_data)
    print "boot interval", bootstrap_confidence(sample_data, delta_conf, num_b)
    print "boot upper", bootstrap_confidence_upper(sample_data, delta_conf, num_b)
    print "value at risk 95%", value_at_risk(sample_data, delta_conf) 
    print "phil lower bound 95%", phil_lower_bnd(sample_data, delta_conf, 1)
    print "phil upper bound 95%", phil_upper_bnd(sample_data, delta_conf, 1)
    print "t-test", ttest_upper_bnd(sample_data, delta_conf)

if __name__=="__main__":
    main()
