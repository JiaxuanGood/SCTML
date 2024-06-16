import numpy as np
import math
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from Inst import *
import sklearn.cluster as clust

def softthres(x,e):
    a=np.maximum(x-e,0)
    b=np.maximum(-1*x-e,0)
    return a-b

def laplacian(X, k=5):
    if(k<len(X)):
        numNeibor = k
    else:
        numNeibor = int(len(X)/2+1)
    findNb = NearestNeighbors(n_neighbors=numNeibor, algorithm='ball_tree')
    findNb.fit(X)
    indices = findNb.kneighbors(X, return_distance=False)
    n = len(X)
    X = normX1(X)
    S = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if(j in set(indices[i])):
                S[i,j] = np.exp(-0.5*math.pow(np.linalg.norm((X[i]-X[j])), 2))
                # S[i,j] = 1
    D = np.zeros((n,n))
    for i in range(n):
        D[i,i] = np.sum(S[i])
    H = D-S
    return H

def laplacian_Y(Y):
    n = np.shape(Y)[1]
    S = pairwise_distances(np.transpose(Y),metric="cosine")
    D = np.zeros((n,n))
    for i in range(n):
        D[i,i] = np.sum(S[i])
    L = D-S
    return L

def getClusterId(data, k):
    kmeans = clust.KMeans(n_clusters=k).fit(data)
    index = []
    for i in range(k):
        index.append(np.argwhere(kmeans.labels_==i).flatten())
    return index

def getClusterId2(data1,data2, k):
    tmp = np.arange(len(data1))
    kmeans = clust.KMeans(n_clusters=k).fit(data2)
    index = []
    for i in range(k):
        index.append(np.append(tmp,np.argwhere(kmeans.labels_==i).flatten()))
    return index

def laplacian_X(clusterid, data, numKNN=5):
    mat = []
    for i in range(len(clusterid)):
        mat.append(laplacian(data[clusterid[i]], k=numKNN))
    return mat

#Lasso is implemented using the accelerated proximal gradient 
def meta(X_org, Xu, Xl,Y,alpha,beta,enta,maxIter,miniLossMargin,numClusters=8,numNeighbors=5):
    """
    X:the confidence score matrix
    Y:label
    alpha: sparsity parameter
    beta: instance correlation parameter
    enta: label correlation parameter
    maxIter: max interation
    """
    # print(np.shape(Xl),np.shape(Xu))
    X = np.vstack((Xl,Xu))
    XTX=np.dot(np.transpose(Xl), Xl)
    XTY=np.dot(np.transpose(Xl), Y)
    #Initialize the w0,w1
    # W_s = np.dot(np.linalg.inv(XTX + enta * np.eye(n_features)),XTY).astype(np.float)
    W_s = np.dot(np.ones(np.shape(XTY)), 0.33)
    W_s_1 = W_s
    # Calculate the similarity distance
    # H = laplacian(X_org, k=10)
    clusterid = getClusterId(X_org, k=numClusters)
    Hs = laplacian_X(clusterid, X_org, numNeighbors)
    H_Y = laplacian_Y(Y)

    iter = 1
    oldloss = 0
    # Colculate Lipschitz constant
    # Lip = math.sqrt( 2 * math.pow(np.linalg.norm(XTX,ord=2),2) + 2* Lip2 )
    Lip2 = 0
    for i in range(numClusters):
        Lip2 = Lip2 + math.pow(np.linalg.norm(beta*np.dot(np.dot(np.transpose(X[clusterid[i]]), Hs[i]), X[clusterid[i]]) ,ord=2), 2)
    Lip = math.sqrt( 2 * math.pow(np.linalg.norm(XTX,ord=2),2) + 
        2 * Lip2 + 
        2 * math.pow(np.linalg.norm(H_Y,ord=2), 2))
    # Initialize b0,b1
    bk=1
    bk_1=1
    # the accelerate proximal gradient
    while iter<=maxIter:
        W_s_k = W_s + np.dot((bk_1 - 1) / bk ,(W_s - W_s_1))
        
        tmp = 0
        for i in range(numClusters):
            tmp = tmp + np.dot( np.dot(np.dot(np.transpose(X[clusterid[i]]), Hs[i]), X[clusterid[i]]), W_s_k)
        Gw_s_k = W_s_k - (1 / Lip) * ((np.dot(XTX , W_s_k) - XTY) + beta * tmp + enta * np.dot(W_s_k, H_Y) )
        

        bk_1 = bk
        bk = (1 + math.sqrt(4 * math.pow(bk,2) + 1)) / 2
        W_s_1 = W_s
        # soft-thresholding operation
        W_s = softthres(Gw_s_k, alpha / Lip)

        a=np.transpose(np.dot(Xl,W_s)-Y)
        b=np.dot(Xl,W_s)-Y

        # Calculate the least squares loss
        predictionLoss=np.trace(np.dot(a,b))
        # Calculate correlation
        # XW = np.dot(X,W_s)
        # correlation=np.trace(np.dot(np.dot(np.transpose(XW), H),XW))
        XW = []
        for i in range(numClusters):
            XW.append(np.dot(X[clusterid[i]],W_s))
        correlation = 0
        for i in range(numClusters):
            correlation = correlation + np.trace(np.dot(np.dot(np.transpose(XW[i]), Hs[i]),XW[i]))
        correlation_Y=np.trace(np.dot(np.dot(W_s, H_Y),np.transpose(W_s)))
        # Calculate sparsity
        sparsity=np.sum(np.sum(np.int64(W_s!=0)))
        # Calculate total loss
        totalloss = predictionLoss + beta * correlation + alpha * sparsity + enta * correlation_Y

        if math.fabs(oldloss-totalloss) <= miniLossMargin:
            break
        elif totalloss <= 0:
            break
        else:
            oldloss = totalloss

        iter = iter + 1

    return W_s
