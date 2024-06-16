from time import time
import numpy as np
from sklearn import svm
from mReadData import *
from mEvaluation import evaluate
from skmultilearn.problem_transform.cc import ClassifierChain
from sklearn.tree import DecisionTreeClassifier as DT
import random
from Inst import *
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.base import BaseEstimator,ClassifierMixin

def randorder(Q):
    return np.array(random.sample(range(Q),Q))

class MSVM(ClassifierMixin):
    def __init__(self, *, param=1):
        self.param = param
    def fit(self,X,y):
        self.output = -1
        if(np.sum(y)==len(y)):
            self.output = 1
        elif(np.sum(y)==0):
            self.output = 0
        else:
            # self.classifier = svm.SVC(probability=True,C=1.0, kernel='rbf',gamma='scale')
            self.classifier = svm.SVC(probability=True, max_iter=1000)
            self.classifier.fit(X,y)
    def predict(self, Xt):
        if(self.output==-1):
            return self.classifier.predict_proba(Xt)[:,1]
        else:
            return np.array([self.output]*len(Xt))

class Baser():
    def __init__(self,X,y):
        self.output = -1
        if(np.sum(y)==len(y)):
            self.output = 1
        elif(np.sum(y)==0):
            self.output = 0
        else:
            # self.classifier = svm.SVC(probability=True,C=1.0, kernel='rbf',gamma='scale')
            self.classifier = svm.SVC(probability=True, max_iter=1000)
            self.classifier.fit(X,y)
    def test(self, Xt):
        if(self.output==-1):
            return self.classifier.predict_proba(Xt)[:,1]
        else:
            return np.array([self.output]*len(Xt))
class BR():
    def __init__(self,X,Y):
        self.baseLearner = []
        self.Q = np.shape(Y)[1]
        for j in range(self.Q):
            singleLearner = svm.SVC(probability=True,C=1.0, kernel='rbf',gamma='scale')
            singleLearner.fit(X,Y[:,j])
            self.baseLearner.append(singleLearner)
    def test(self,Xt):
        prediction = []
        for j in range(self.Q):
            prediction_a = self.baseLearner[j].predict_proba(Xt)[:,1]
            prediction.append(prediction_a)
        return np.array(np.transpose(prediction))
class CC():
    def __init__(self,X,Y,order=[]):
        self.baseLearner = []
        self.num_label = np.shape(Y)[1]
        if(len(order)==0):
            self.order = randorder(self.num_label)
        else:
            self.order = order
        X_train = np.array(X)
        # print(self.order)
        for j in self.order:
            singleLearner = svm.SVC(probability=True,C=1.0, kernel='rbf',gamma='scale')
            singleLearner.fit(X_train,Y[:,j])
            self.baseLearner.append(singleLearner)
            X_train = np.hstack((X_train, Y[:,[j]]))
    def test(self,Xt):
        Xt_train = np.array(Xt)
        prediction= [[] for _ in range(self.num_label)]
        for i in range(len(self.order)):
            j = self.order[i]
            prediction_a = self.baseLearner[i].predict_proba(Xt_train)[:,1]
            prediction[j] = prediction_a
            prediction_a = np.reshape(prediction_a, (-1, 1))
            Xt_train = np.hstack((Xt_train, prediction_a))
        return np.transpose(prediction)
class CCE():
    def __init__(self,X,Y,idxs,order=[]):
        self.baseLearner = []
        self.num_label = np.shape(Y)[1]
        if(len(order)==0):
            self.order = randorder(self.num_label)
        else:
            self.order = order
        X_train = np.array(X)
        # print(self.order)
        for j in self.order:
            idx = np.argwhere(idxs[j]).flatten()
            # singleLearner = svm.SVC(probability=True,C=1.0, kernel='rbf',gamma='scale')
            # singleLearner.fit(X_train[idx],Y[:,j][idx])
            singleLearner = Baser(X_train[idx],Y[:,j][idx])
            self.baseLearner.append(singleLearner)
            X_train = np.hstack((X_train, Y[:,[j]]))
    def test(self,Xt):
        Xt_train = np.array(Xt)
        prediction= [[] for _ in range(self.num_label)]
        for i in range(len(self.order)):
            j = self.order[i]
            # prediction_a = self.baseLearner[i].predict_proba(Xt_train)[:,1]
            prediction_a = self.baseLearner[i].test(Xt_train)
            prediction[j] = prediction_a
            prediction_a = np.reshape(prediction_a, (-1, 1))
            Xt_train = np.hstack((Xt_train, prediction_a))
        return np.transpose(prediction)
class CCM():
    def __init__(self,X,Y,order=[]):
        self.baseLearner = []
        self.num_label = np.shape(Y)[1]
        if(len(order)==0):
            self.order = randorder(self.num_label)
        else:
            self.order = order
        X_train = np.array(X)
        # print(self.order)
        for j in self.order:
            # singleLearner = svm.SVC(probability=True,C=1.0, kernel='rbf',gamma='scale')
            # singleLearner.fit(X_train[idx],Y[:,j][idx])
            singleLearner = Baser(X_train,Y[:,j])
            self.baseLearner.append(singleLearner)
            X_train = np.hstack((X_train, Y[:,[j]]))
    def predict(self,Xt):
        Xt_train = np.array(Xt)
        prediction= [[] for _ in range(self.num_label)]
        for i in range(len(self.order)):
            j = self.order[i]
            # prediction_a = self.baseLearner[i].predict_proba(Xt_train)[:,1]
            prediction_a = self.baseLearner[i].test(Xt_train)
            prediction[j] = prediction_a
            prediction_a = np.reshape(prediction_a, (-1, 1))
            Xt_train = np.hstack((Xt_train, prediction_a))
        return np.transpose(prediction)
class WCC():
    def __init__(self,order=[]):
        self.baseLearner = []
        self.num_label = 0
        self.order = order
    def train(self,X,Y,distribution=[]):
        X_train = np.array(X)
        self.num_label = np.shape(Y)[1]
        if(len(self.order)==0):
            self.order = randorder(self.num_label)
        for j in self.order:
            singleLearner = svm.SVC(probability=True,C=1.0, kernel='rbf',gamma='scale')
            if(len(distribution)==0):
                singleLearner.fit(X_train,Y[:,j])
            else:
                singleLearner.fit(X_train,Y[:,j],distribution[j])
            self.baseLearner.append(singleLearner)
            X_train = np.hstack((X_train, Y[:,[j]]))
    def test(self,Xt):
        Xt_train = np.array(Xt)
        prediction= [[] for _ in range(self.num_label)]
        for i in range(len(self.order)):
            j = self.order[i]
            prediction_a = self.baseLearner[i].predict_proba(Xt_train)[:,1]
            prediction[j] = prediction_a
            prediction_a = np.reshape(prediction_a, (-1, 1))
            Xt_train = np.hstack((Xt_train, prediction_a))
        return np.transpose(prediction)

if __name__=="__main__":
    '''Ablation Study'''
    def fill1(Y):
        Y = np.array(Y)
        for j in range(np.shape(Y)[1]):
            if(np.sum(Y[:,j])==0):
                Y[0][j] = 1
        return Y

    numBase = 10

    datasnames = ["3Sources_bbc1000","3Sources_guardian1000","3Sources_inter3000","3Sources_reuters1000","Birds","CAL500","CHD_49","Enron","Flags","Foodtruck",
        "GnegativeGO","GpositiveGO","Image","Langlog","Medical","PlantGO","Scene","Slashdot","Chemistry","Chess",
        "Coffee","VirusGO","Yeast","Yelp","Corel5k","Philosophy"]
    rd = ReadData(datas=datasnames,genpath='arff/')
    for dataIdx in range(4,5):
        print(dataIdx)
        # X,Y,Xt,Yt = rd.readData(dataIdx)
        k_fold,X_all,Y_all = rd.readData_CV(dataIdx)
        for train, test in k_fold.split(X_all, Y_all):
            X = X_all[train]
            Y = Y_all[train]
            Xt = X_all[test]
            Yt = Y_all[test]
            Y = fill1(Y)
            start_time = time()
            prediction = []
            blearner = CC()
            blearner.train(X, Y)
            prediction = blearner.test(Xt)
            resolveResult(datasnames[dataIdx], 'CC', evaluate(prediction, Yt), (time()-start_time))
