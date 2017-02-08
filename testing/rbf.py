'''
Created on Feb 8, 2017

@author: dsbrown
'''

#I guess I could fit this to the true reward and then use the ground truth basis function weights from the fitting 
# to compare with rewards learned through BIRL

#use DMP code to fit weights to a bunch of samples along position with the true mdp reward -1 everywhere?

import numpy as np
import matplotlib.pyplot as plt

def predictRBF(s_query, c, h, weights):
    phi = np.exp(-h*(s_query-c)**2)
    return phi.dot(weights)

numCenters = 10
x_min = -1.2
x_max = 0.6
c = np.linspace(x_min,x_max,numCenters) #centers
h = 100*np.ones(numCenters) #widths (bigger is smaller variance)
c = np.asarray(c)
h = np.asarray(h)

weights = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

phaseRange = np.linspace(x_min,x_max,1000) #for visualizing basis functions
#plot basis functions as function of s
plt.figure(1)
for bfn in range(numCenters):
    plt.plot(phaseRange, np.exp(-h[bfn]*(phaseRange - c[bfn])**2))

plt.figure(2)
preds = [predictRBF(s, c, h, weights) for s in phaseRange]
plt.plot(phaseRange, preds)

plt.show()


