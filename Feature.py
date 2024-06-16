import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from saveRst import *
import math
from Inst import *

def ent(p):
    if(p==0):
        return 0
    else:
        return p*np.log2(p)

def entropy(y):
    p1 = sum(y)/len(y)
    p0 = 1-p1
    return ent(p0)+ent(p1)

def joint_entropy(x,y):
    n = len(x)
    p11 = sum(x+y==2)/n
    p00 = sum(x+y==0)/n
    p01 = sum((1-x)*y==1)/n
    p10 = sum(x*(1-y)==1)/n
    return ent(p00)+ent(p01)+ent(p10)+ent(p11)

def mutual_information(x,y):
    return entropy(x)+entropy(y)-joint_entropy(x,y)

def Gaussian(x,y,bandwidth=1):
    return np.exp(-math.pow(np.linalg.norm((x-y)), 2)/(2*bandwidth))

def pairwise(X,Y,metric='',bandwidth=1):
    n1 = len(X)
    n2 = len(Y)
    mat = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            if(metric=='Gaussian'):
                mat[i,j] = Gaussian(X[i],Y[j],bandwidth)
            elif(metric=='Entropy'):
                mat[i,j] = joint_entropy(X[i],Y[j])
            elif(metric=='MIN'):
                mat[i,j] = mutual_information(X[i],Y[j])
            else:
                print('error')
    return mat

def getidxset_base(X, Y):
    X2 = np.array(normX1(X)>0.5)*1
    if(len(Y)<len(X)):
        X2 = X2[:len(Y)]
    pG = pairwise(np.array(np.transpose(X2)),np.transpose(Y),'MIN')
    id = []
    for i in range(len(Y[0])):
        id.append(np.argmax(pG[:,i]))
    return np.int16(np.unique(id))

def getidxset_dask(X, Y, numbase):
    num_feat = np.shape(X)[1]
    simlarity = pairwise_distances(np.transpose(X),metric="cosine")
    X2 = np.array(X)
    if(len(Y)<len(X)):
        X2 = X2[:len(Y)]
    sims = pairwise_distances(np.array(np.transpose(X2)),np.transpose(Y),metric="cosine")
    entropy = np.average(sims,1)

    '''initial subsets & entropy set'''
    idxset = [[] for _ in range(numbase)]
    setent = np.zeros(numbase)
    for i in range(numbase):
        idxset[i].append(i)
        setent[i] = entropy[i]

    for i in range(numbase,num_feat): # search the i-th feature
        j = np.argmin(setent) # the want set via entropy
        simtmp = np.ones(numbase)
        for t in range(numbase):
            simtmp[t] = np.min(simlarity[i,idxset[t]])
        k = np.argmin(simtmp) # the want set via similarity

        z = [j,k][np.random.randint(2)]
        idxset[z].append(i)
        setent[z] += entropy[i]
        # idxset[j].append(i)
        # setent[j] += entropy[i]
        # idxset[k].append(i)
        # setent[k] += entropy[i]
    return idxset

def getidxset_kmeans(X, Y, numbase):
    base = getidxset_base(X,Y)
    locs = kmeans3(np.transpose(X))
    idxset = []
    for i in range(numbase):
        idxset.append(np.argwhere(locs==i).flatten().tolist())
        idxset[i] = np.unique(np.append(idxset[i],base)).tolist()
        idxset[i] = np.int16(idxset[i])
    return idxset

def getidxset_kmeans2(X, Y, numbase):
    base = getidxset_base(X,Y)
    locs = kmeans3(np.transpose(X), base)
    idxset = []
    for i in range(numbase):
        idxset.append(np.argwhere(locs==i).flatten().tolist())
        idxset[i] = np.unique(np.append(idxset[i],base)).tolist()
    return idxset

def getidxset0(X, numbase):
    num_feat = np.shape(X)[1]
    idxset = [[] for _ in range(numbase)]
    case = num_feat/numbase
    for i in range(numbase):
        idxset[i] = np.arange(int(case*i),int(case*(i+1)))
    return idxset

def kmeans3(data, base=[]):
    n = len(data)
    centers = []
    if(len(base)==0):
        centers_id = []
        simlarity = pairwise_distances(data,metric="cosine")
        dist_0 = 0
        for i in range(n-2):
            for j in range(i+1,n-1):
                for k in range(j+1,n):
                    dist_1 = simlarity[i,j] + simlarity[i,k] + simlarity[j,k]
                    if(dist_1>dist_0):
                        centers_id = [i,j,k]
                        dist_0 = dist_1
        for j in range(3):
            centers.append(data[centers_id[j]])
    else:
        for j in range(3):
            centers.append(np.sum(data[base],0)/len(base))
            # print(np.shape(data),np.shape(data[base]),np.shape(centers))
    locs = np.zeros(n)
    for i in range(n):
        tmp_dist = []
        for j in range(3):
            tmp_dist.append(pairwise_distances([data[i]],[centers[j]],metric="cosine")[0])
            # tmp_dist.append(pairwise([data[i]],[centers[j]],metric='Gaussian')[0])
        locs[i] = np.argmax(tmp_dist)
    return locs

def getsubX(X, idxset):
    subXs = []
    data = np.transpose(X)
    for _ in idxset:
        subXs.append(np.transpose(data[_]))
    return subXs

if __name__=="__main__":
    # x = np.random.random((100,30))
    # simlarity = pairwise_distances([x[0]],[x[1]],metric="cosine")[0]
    # centers_id = np.random.randint(0,3,10)
    # print(x)
    # print(centers_id)
    # print(np.argwhere(centers_id==1).flatten())
    # print(np.average(x[np.argwhere(centers_id==1).flatten()],0))
    # locs = kmeans3(x)
    # print(locs)
    # print(sum(locs==0),sum(locs==1),sum(locs==2))

    x = np.random.random((100,30))
    y = np.random.random((20,10))
    y = np.array(y>np.average(np.average(y)))*1
    numbase = 3
    # basic = getidxset_base(x,y)
    # print(basic)

    # idset = getidxset_dask(x,y,numbase)
    # print(idset)
    # xs = getsubX(x, idset)
    # for i in range(numbase):
    #     print(np.shape(idset[i]),np.shape(xs[i]))
    
    # idset = getidxset_kmeans2(x,y,numbase)
    idset = getidxset0(x,numbase)
    print(idset)
    xs = getsubX(x, idset)
    for i in range(numbase):
        print(np.shape(idset[i]),np.shape(xs[i]))
    