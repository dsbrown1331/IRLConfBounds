'''
Created on Feb 10, 2017

@author: dsbrown
'''

import numpy as np
import scipy as scp
import scipy.stats as st
from bounds import bound_methods

data = [325, 325,   334,   339,   356,   356,   359,   359,   363,
364,   364,   366,   369,   370,   373,   373,   374,   375,
389,   392,   393,   394,   397,   402,   403,   424]
#percentile
p = 0.75  
#confidence level
alpha = 0.08
num_samples = len(data)
print p * num_samples
bin_mean = num_samples * p
bin_var = num_samples * p * (1 - p)
bin_std = np.sqrt(bin_var)

print "upper and lower bounds"
z_upper = st.norm.ppf((1-alpha) + alpha/2.0)
z_lower = st.norm.ppf(alpha/2.0)
print z_upper, z_lower
lower_order_idx = int(np.floor(z_lower * bin_std + bin_mean + 0.5))
upper_order_idx = int(np.ceil(z_upper * bin_std + bin_mean + 0.5))
print lower_order_idx, upper_order_idx

#double check math 
print st.norm.cdf((upper_order_idx - 0.5 - bin_mean)/bin_std) - st.norm.cdf((lower_order_idx - 0.5 - bin_mean)/bin_std)
print (data[lower_order_idx-1], data[upper_order_idx-1])

print "upper bound"
z_upper = st.norm.ppf(1-alpha)
print z_upper
upper_order_idx = int(np.ceil(z_upper * bin_std + bin_mean + 0.5))
print upper_order_idx

#double check math 
print st.norm.cdf((upper_order_idx - 0.5 - bin_mean)/bin_std)
print data[upper_order_idx-1]
print bound_methods.percentile_confidence_upper_bnd(data, p, 1-alpha)