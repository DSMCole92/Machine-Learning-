import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def distance(P,Q):
    return ( (P[0] - Q[0])**2 + (P[1] - Q[1])**2 )**0.5

def assign(pts, ctrs):
      
    clusterAssign = []
    for i in range(0,len(pts)):
        
        curDist = distance(pts[i,:], ctrs[0,:])
        curClass = 0
        
        for j in range(1, len(ctrs)):    
            tempDist = distance(pts[i,:],ctrs[j,:])
            if tempDist < curDist:
                curClass = j
                curDist = tempDist    
        
        clusterAssign.append(curClass)
        
    return np.array(clusterAssign)

def newCenters(pts, clAssign, K):
    
    xCtr = []
    yCtr = []
    
    for i in range(0, K):
        sel = (clAssign == i)
        selX = pts[sel,0]
        selY = pts[sel,1]
        xCtr.append(selX.mean())
        yCtr.append(selY.mean())
		
    xCtr = np.array(xCtr).reshape(-1,1)
    yCtr = np.array(yCtr).reshape(-1,1)
		
    ctrs = np.hstack([xCtr, yCtr])
    
    return ctrs

def icDist(pts, ctrs, clAssign):
    
    totalDist = 0
    
    for i in range(0, len(pts)):
        cl = clAssign[i]
        totalDist = totalDist + distance(pts[i,:], ctrs[cl,:])
        
    return totalDist

def kMeans(pts, K):
    
    # Choose Random Centers
    sel = np.random.choice(range(0,len(pts)), K, replace=False)
    centers = pts[sel,:]

    clusters = assign(pts,centers) 
    centers = newCenters(pts, clusters, K)
    curDist = icDist(pts, centers, clusters)
    
    done = False    
    
    while done == False:
        clusters = assign(pts,centers) 
        centers = newCenters(pts, clusters, K)
        newDist = icDist(pts, centers, clusters)
        
        if(curDist - newDist < 0.0001):
            done = True
        curDist = newDist
    
    plt.scatter(pts[:,0],pts[:,1], c = clusters)
    plt.show()