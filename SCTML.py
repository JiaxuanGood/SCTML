from mReadData import ReadData
from mEvaluation import evaluate
from Base import *
from Meta import meta
from saveRst import *
from Feature import *
from Inst import *

if __name__ == '__main__':
    datasnames = ["Birds","Enron","Langlog","Medical","Scene","VirusGO","Yeast","Yelp","HumanGO","Tmc2007_500"]
    rd = ReadData(datas=datasnames,genpath='arff/')
    numdata = 1
    n_fold = 5
    for dataIdx in range(numdata):
        print(n_fold,dataIdx)
        tmp_rst = np.zeros(13)
        k_fold,X,Y = rd.readData_CV(dataIdx,n_fold)
        for train, test in k_fold.split(X, Y):
            Xl,Yl,Xu,Yu = datasplit(X[train], Y[train], 1/(n_fold-1))
            Xt,Yt = X[test],Y[test]
            print(np.shape(Xl),np.shape(Xu),np.shape(Xt),np.shape(Yl),np.shape(Yu),np.shape(Yt))

            START_TIME = time()
            numbase = 3
            numlabel = np.shape(Y)[1]
            idxset = getidxset_dask(X,Yl,numbase)
            Xls = getsubX(Xl, idxset)
            Xus = getsubX(Xu, idxset)
            Xts = getsubX(Xt, idxset)
            tau0 = 0.7
            '''Learner Generation'''
            baseLearners = [[[] for _ in range(numlabel)] for _ in range(numbase)]
            for i in range(numbase):
                for j in range(numlabel):
                    baseLearners[i][j] = Baser(Xls[i],Yl[:,j])
            '''Data Expansion'''
            locfit = [[[] for _ in range(numlabel)] for _ in range(numbase)]
            fakelabel = [[[] for _ in range(numlabel)] for _ in range(numbase)]
            for i in range(numbase):#numbase
                for j in range(numlabel):
                    tau = tau0
                    tmp = baseLearners[i][j].test(Xus[i])
                    fakelabel[i][j] = np.round(tmp)
                    tmp = np.abs(tmp*2-1)
                    locfit[i][j] = np.array(tmp>tau)*1
                    if(sum(locfit[i][j])<len(Xt)*(1-tau0)):
                        tau = np.sort(tmp)[int(tau0*len(tmp))]
                        locfit[i][j] = np.array(tmp>tau)*1
            '''Interactive annotation'''
            locfit2 = [[[] for _ in range(numlabel)] for _ in range(numbase)]
            for i in range(numbase):#numbase
                p = (i+1)%3
                q = (i+2)%3
                for j in range(numlabel):
                    locfit2[i][j] = np.array(fakelabel[p][j]==fakelabel[q][j])*1
            '''Learner Enhance'''
            def getarr(A):
                for i in range(len(A)):
                    A[i] = np.array(A[i])
                return A
            mLearners = [[] for _ in range(numbase)]
            for i in range(numbase):
                mLearners[i] = CCE(np.vstack((Xls[i],Xus[i])),np.vstack((Yl,np.transpose(np.array(fakelabel[i])))),np.array(locfit[i])*np.array(locfit2[i]))
            new_Xl = []
            new_Xu = []
            for i in range(numbase):
                new_Xl.append(mLearners[i].test(Xls[i]))
                new_Xu.append(mLearners[i].test(Xus[i]))
            new_Xl = np.hstack(tuple(new_Xl))
            new_Xu = np.hstack(tuple(new_Xu))
            '''
            alpha: L1
            beta: manifold
            enta: label correlation
            '''
            W = meta(np.vstack((Xl,Xu)), new_Xu, new_Xl, Yl, 0.1, 0.01, 0.01, 200, 0.0001, 8) # SCTML

            MID_TIME = time()
            new_Xt = []
            for i in range(numbase):
                new_Xt.append(mLearners[i].test(Xts[i]))
            new_Xt = np.hstack(tuple(new_Xt))
            Pt = np.dot(new_Xt, W)
            tmp = evaluate(Pt, Yt)
            tmp.append(MID_TIME-START_TIME)
            tmp.append(time()-MID_TIME)
            tmp_rst += np.array(tmp)
        resolveResult(datasnames[dataIdx], 'SCT', tmp_rst/n_fold)