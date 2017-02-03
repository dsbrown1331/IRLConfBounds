# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 22:03:35 2017

@author: daniel
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_nav_world(size, features, demonstrations):
    # Make a 9x9 grid...
    nrows, ncols = size, size
    image = np.zeros((nrows,ncols))
    white_f = [1,0,0,0,0,0]
    red_f = [0,1,0,0,0,0]
    blue_f = [0,0,1,0,0,0]
    yellow_f = [0,0,0,1,0,0]
    green_f = [0,0,0,0,1,0]
    purple_f = [0,0,0,0,0,1]
    # Set every other cell to a random number (this would be your data)
    for f in features:
        #print f
        #print features[f]
        c,r = f
        if features[f] == white_f: 
            col = 8
        elif features[f] == red_f:
            col = 0
        elif features[f] == blue_f:
            col = 1
        elif features[f] == yellow_f:
            col = 5
        elif features[f] == green_f:
            col = 2
        elif features[f] == purple_f:
            col = 3
            
        image[r,c] = col
    print "image", image
        
    #image[0,0] = 0 #red
    #image[0,1] = 1 #blue
    #image[0,2] = 2 #green
    #image[0,3] = 3 #purple
    #image[0,4] = 4 #orange
    #image[0,5] = 5 #yellow
    #image[0,6] = 6 #brown
    #image[0,7] = 7 #pink
    #image[0,8] = 8 #grey

    #plot demonstration paths
    for path in demonstrations:
        #given a path extract x,y coords
        pathx = []
        pathy = []

        #add an offset so the shown path goes through the middle of the squares
        offset = 0.5
        for x,y in path:
            pathx.append(x + offset)
            pathy.append(y + offset)

        plt.plot(pathx, pathy,'k-',linewidth=4)   
        #plot circle at start
        plt.plot(pathx[0], pathy[0],'ko',linewidth=4)
        plt.plot(pathx[-1], pathy[-1],'kD',linewidth=4)

    f = plt.pcolor(image, cmap = 'Set1')
    f.axes.get_xaxis().set_visible(False)
    f.axes.get_yaxis().set_visible(False)
    plt.gca().set_xlim((0,size))
    plt.gca().set_ylim((0,size))
    plt.show()