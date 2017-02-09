'''
Created on Feb 8, 2017

@author: dsbrown
'''

#I guess I could fit this to the true reward and then use the ground truth basis function weights from the fitting 
# to compare with rewards learned through BIRL

#use DMP code to fit weights to a bunch of samples along position with the true mdp reward -1 everywhere?
#TODO trying to get multiple dims working

import numpy as np
import matplotlib.pyplot as plt

#uses gaussian RBFs
def predictRBF(s_query, c, h, weights):
    phi = np.exp(-h*(s_query-c)**2)
    return phi.dot(weights)

def predictRBFndim(query, c, h, weights):
    #calc dists to centers
    print query - c
    dists = np.linalg.norm(query-c, axis=1)
    print "dists", dists
    print -h*(np.linalg.norm(query-c, axis=1))**2
    phi = np.exp(-h*(np.linalg.norm(query-c, axis=1))**2)
    return phi.dot(weights)

numCentersX = 3 
x_min = -1.2
x_max = 0.6
numCentersY = 3
y_min = -0.07
y_max = 0.07
numRBFs = numCentersX * numCentersY
cX = np.linspace(x_min,x_max, numCentersX) #centers
cY = np.linspace(y_min, y_max, numCentersY)
h = 100*np.ones(numRBFs) #widths (bigger is smaller variance)

weights = [0,1,2,3,4,5,6,7,8]
c = np.array([[x, y] for x in cX for y in cY])
h = np.array(h)
print h
print c
print predictRBFndim([1,1], c, h, weights)
print bunny




phaseRange = np.linspace(x_min,x_max,1000) #for visualizing basis functions
#plot basis functions as function of s
plt.figure(1)
for bfn in range(numCenters):
    plt.plot(phaseRange, np.exp(-h[bfn]*(phaseRange - c[bfn])**2))

plt.figure(2)
preds = [predictRBF(s, c, h, weights) for s in phaseRange]
plt.plot(phaseRange, preds)

plt.show()


