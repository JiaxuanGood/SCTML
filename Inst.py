import numpy as np
from sklearn import svm

def datasplit131(X, Y, Xt, Yt):
    X_all = np.vstack((X,Xt))
    Y_all = np.vstack((Y,Yt))
    Xlu,Ylu,Xt,Yt = datasplit(X_all, Y_all, 0.8)
    Xl,Yl,Xu,Yu = datasplit(Xlu, Ylu, 0.25)
    X = np.vstack((Xl,Xu))
    return Xl,Yl,Xu,Yu,Xt,Yt

def datasplit(X, Y, ratio=0.2):
    n = len(X)
    u = int(n*ratio)
    Xl = X[:u]
    Yl = Y[:u]
    Xu = X[u:]
    Yu = Y[u:]
    return Xl,Yl,Xu,Yu
def normX(X,Xt):
    n = len(X)
    X2 = np.transpose(np.vstack((X,Xt)))
    for i in range(len(X2)):
        m1 = np.min(X2[i])
        m2 = np.max(X2[i])
        if(m1==m2):
            X2[i] = 1
        else:
            X2[i] = (X2[i]-m1)/(m2-m1)
    X2 = np.transpose(X2)
    return X2[:n],X2[n:]
def normX1(X2):
    X2 = np.transpose(X2)
    for i in range(len(X2)):
        m1 = np.min(X2[i])
        m2 = np.max(X2[i])
        if(m1==m2):
            X2[i] = 1
        else:
            X2[i] = (X2[i]-m1)/(m2-m1)
    return np.transpose(X2)
def fill1(Y):
    Y = np.array(Y)
    for j in range(np.shape(Y)[1]):
        if(np.sum(Y[:,j])==0):
            Y[0][j] = 1
    for j in range(np.shape(Y)[1]):
        if(np.sum(1-Y[:,j])==0):
            Y[0][j] = 0
    return Y