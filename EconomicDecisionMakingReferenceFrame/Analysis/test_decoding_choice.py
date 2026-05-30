import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC,SVR
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.decomposition import PCA 
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve, auc
import os,sys
from pathlib import Path
import pandas as pd

dirName = sys.argv[1]


def main():
    for root, dirs, files in os.walk("/gpfsnyu/scratch/jtg374/psychrnn/savedForHPC/"+dirName, topdown=False):
        for name in dirs:
            dirPath = os.path.join(root,name)
            # --------behavior -----------------------
            # if Path(os.path.join(dirPath,"decodingChoice.npz")).exists():
            #     continue
            print(dirPath)

            x,trial_params,model_state,choice12,choiceAB,choiceLR,qAs,qBs,seqAB = importAndPreprocess(
                dirPath=dirPath,
                activityFileName='activitityTest.npz')

            time = np.arange(0,4000,10)
            cv=5
            accuracyLR = decoding_choice_inPlace_cv(model_state,choiceLR,cv=cv)
            accuracyAB = decoding_choice_inPlace_cv(model_state,choiceAB,cv=cv)
            accuracy12 = decoding_choice_inPlace_cv(model_state,choice12,cv=cv)

            ymin=.4
            ymax=1
            # plt.vlines([50,100,150,200,250,300],ymin,ymax,linestyles='dashed',label='_nolegend_')
            plt.fill_between([500,1000],0,1,color='gold',alpha=0.3,edgecolor=None)
            plt.fill_between([1500,2000],0,1,color='gold',alpha=0.3,edgecolor=None)
            plt.fill_between([2500,3000],0,1,color='cadetblue',alpha=0.3,edgecolor=None)
            plt.errorbar(time,np.mean(accuracyLR,axis=1),np.std(accuracyLR,axis=1)/np.sqrt(cv),label='LR')
            plt.errorbar(time,np.mean(accuracyAB,axis=1),np.std(accuracyAB,axis=1)/np.sqrt(cv),label='juice')
            plt.errorbar(time,np.mean(accuracy12,axis=1),np.std(accuracy12,axis=1)/np.sqrt(cv),label='order')

            plt.legend()
            plt.xlabel('time (ms)')
            plt.ylabel('accuracy')
            plt.ylim(ymin,ymax)

            np.savez(os.path.join(dirPath,"decodingChoice.npz"),accuracy12=accuracy12,accuracyAB=accuracyAB,accuracyLR=accuracyLR)
            plt.savefig(os.path.join(dirPath,"decodingChoice.pdf"))

            plt.close('all')

def main_bak():
    for root, dirs, files in os.walk("/gpfsnyu/scratch/jtg374/psychrnn/savedForHPC/"+dirName, topdown=False):
        for name in dirs:
            dirPath = os.path.join(root,name)
            print(dirPath)
            # --------behavior -----------------------
            with np.load(os.path.join(dirPath,'activitityTestGrid.npz'),allow_pickle=True) as f:
                x = f['x']
                trial_params = f['trial_params']
                model_output = f['model_output']
                model_state = f['model_state']
                mask = f['mask']
            
            temp = np.mean(mask * model_output,1)
            choiceLR = temp[:,1]>temp[:,0]
            choiceLR = choiceLR*2-1 # pos right high, neg left high
            
            choiceFrame = [trial_params[i]['choiceFrame'] for i in range(len(trial_params))]
            
            
            locAB = [(1 if trial_params[i]['locAB']=='AB' or trial_params[i]['locAB']=='12' else -1) for i in range(len(trial_params))]
            loc12 = locAB
            seqAB = [(1 if trial_params[i]['seqAB']=='AB' else -1) for i in range(len(trial_params))]
            
            choiceAB = np.array([(choiceLR[i] * locAB[i] if choiceFrame[i]=='juice' else choiceLR[i] * locAB[i] * seqAB[i]) for i in range(len(trial_params)) ])  # pos A neg B
            choice12 = np.array([(choiceAB[i] * seqAB[i] if choiceFrame[i]=='juice' else choiceLR[i] * locAB[i]) for i in range(len(trial_params)) ])  # pos A neg B



            nT = x.shape[1]
            accuracyLR = np.zeros(nT)
            accuracyAB = np.zeros(nT)
            accuracy12 = np.zeros(nT)

            for iT in range(nT):
                Xt = model_state[:,iT,:].squeeze()
                cv = RepeatedStratifiedKFold(n_splits=5,n_repeats=10)
                scoreLR = cross_val_score(LinearDiscriminantAnalysis(),Xt,choiceLR,scoring='accuracy',cv=cv)
                scoreAB = cross_val_score(LinearDiscriminantAnalysis(),Xt,choiceAB,scoring='accuracy',cv=cv)
                score12 = cross_val_score(LinearDiscriminantAnalysis(),Xt,choice12,scoring='accuracy',cv=cv)
                accuracyLR[iT] = np.mean(scoreLR)
                accuracyAB[iT] = np.mean(scoreAB)
                accuracy12[iT] = np.mean(score12)

            plt.figure(dpi=200)
            plt.plot(accuracyLR)
            plt.plot(accuracyAB)
            plt.plot(accuracy12)
            plt.legend(['space','juice','order'])
            plt.xlabel('time')
            plt.ylabel('accuracy')

            np.savez(os.path.join(dirPath,"decodingChoice.npz"),accuracy12=accuracy12,accuracyAB=accuracyAB,accuracyLR=accuracyLR)
            plt.savefig(os.path.join(dirPath,"decodingChoice.pdf"))    

def participantRatio(X):
    M,N = X.shape
    X = X-np.mean(X,axis=0)
    C = (X.T @ X) / (M-1)
    PR = np.trace(C)**2/np.sum(C**2)
    return PR

def participantRatioPca(X):
    pcaObj = PCA().fit(X)
    lambdas = pcaObj.explained_variance_
    PR = np.sum(lambdas)**2 / np.sum(lambdas**2)
    return PR

def offer1Encoding(Y,trial_params):
    K,N = Y.shape
    seqAB = np.array([trial_params[i]['seqAB']for i in range(K)])
    qAs = np.array([trial_params[i]['qA']for i in range(K)])
    qBs = np.array([trial_params[i]['qB']for i in range(K)])
    modelA = LinearRegression().fit(qAs[seqAB=='AB'].reshape(-1,1),Y[seqAB=='AB',:])
    modelB = LinearRegression().fit(qBs[seqAB=='BA'].reshape(-1,1),Y[seqAB=='BA',:])
    
    return modelA,modelB


def angle_between_axis(v1,v2):
    inner_product = np.dot(v1,v2)
    cosDist = inner_product/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.arccos(cosDist)

def distanceToLinearRepresentation(A,B,Y):
    # Linear Model Z ~ AX + B
    N,K = Y.shape
    B = np.repeat(B,K).reshape(N,K)
    x_hat = np.linalg.lstsq(A,Y-B,rcond=None)[0] # shape (1,K)
    y_hat = A@x_hat + B
    R = (Y-y_hat)
    totalError = np.diag( R.T @ np.eye(N) @ R)
    MSTR = np.mean(totalError)
    return MSTR

def overlap_between_representations(linearModel,Y,Z):
    # Linear Model Y ~ AX + B
    N,K = Y.shape
    A = linearModel.coef_ # shape: (N,1)
    B = linearModel.intercept_ # of shape (N,)
    MSTR_self = distanceToLinearRepresentation(A,B,Y)
    MSTR_cross = distanceToLinearRepresentation(A,B,Z)
    return MSTR_cross/MSTR_self    

def decoding_choice_inPlace_cv(model_state,choice,estimator=LinearDiscriminantAnalysis(),cv=5):
    nT = model_state.shape[1]
    accuracy = np.zeros((nT,cv))
    for iT in range(nT):
        Xt = model_state[:,iT,:]
        accuracy[iT,:] = cross_val_score(estimator,Xt,choice,cv=cv)
    return accuracy


def decoding_value_acrossTime(model_state,value,estimator=SVR()):
    nT = model_state.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(model_state,value,test_size=0.5,stratify=value)
    accuracy = np.zeros((nT,nT))
    for iT_train in range(nT):
        estimator.fit(X_train[:,iT_train,:],y_train)
        for iT_test in range(nT):
            accuracy[iT_train,iT_test] = estimator.score(X_test[:,iT_test,:],y_test)
    return accuracy


def decoding_acrossVariableAndTime(model_state,var1,var2,t1,t2,estimator=LinearDiscriminantAnalysis()):
    pass
    # X_train, X_test, y_train, y_test,z_train,z_test = train_test_split(model_state,var1,var2,test_size=0.5,stratify=var1)
    estimator.fit(model_state[:,t1,:],var1)
    accuracy = estimator.score(model_state[:,t2,:],var2)

def offer2encodingDependOnOffer1(Y,q1,q2):
    # split AB trials before hand
    K,N = Y.shape
    uniqueQ1 = np.unique(q1)
    nC = len(uniqueQ1)
    mag = np.zeros(nC)
    for ii in range(nC):
        idx = np.flatnonzero(q1==uniqueQ1[ii])
        Y0 = np.mean(Y[(q1==uniqueQ1[ii])&(q2==0),:],axis=0)
        Y0 = np.tile(Y0,len(idx)).reshape(len(idx),N)
        coeff = LinearRegression().fit(q2[idx].reshape(-1,1),Y[idx,:]-Y0).coef_.reshape(-1)
        mag[ii] = np.linalg.norm(coeff)

    return uniqueQ1,mag


def importAndPreprocess(dirPath,activityFileName):
    import os
    import sys

    
    with np.load(os.path.join(dirPath,activityFileName),allow_pickle=True) as f:
        x = f['x']
        trial_params = f['trial_params']
        model_output = f['model_output']
        model_state = f['model_state']
        mask = f.get('mask', None)
    
    if mask is None:
        temp = np.mean(model_output[:,300:,:],1)
    else: 
        temp = np.mean(mask * model_output,1)
    choiceLR = temp[:,1]>temp[:,0]
    choiceLR = choiceLR*2-1 # pos right high, neg left high
    
    choiceFrame = [trial_params[i]['choiceFrame'] for i in range(len(trial_params))]
    
    
    locAB = [(1 if trial_params[i]['locAB']=='AB' or trial_params[i]['locAB']=='12' else -1) for i in range(len(trial_params))]
    loc12 = locAB
    seqAB = [(1 if trial_params[i]['seqAB']=='AB' else -1) for i in range(len(trial_params))]
    
    choiceAB = np.array([(choiceLR[i] * locAB[i] if choiceFrame[i]=='juice' else choiceLR[i] * locAB[i] * seqAB[i]) for i in range(len(trial_params)) ])  # pos B neg A
    choice12 = np.array([(choiceAB[i] * seqAB[i] if choiceFrame[i]=='juice' else choiceLR[i] * locAB[i]) for i in range(len(trial_params)) ])  # pos 2 neg 1
    choiceAB = np.array(['B' if choiceAB[i]>0 else 'A' for i in range(len(trial_params))])
    choice12 = np.array(['2' if choice12[i]>0 else '1' for i in range(len(trial_params))])
    
    qAs = np.array([trial_params[i]['qA'] for i in range(len(trial_params))])
    qBs = np.array([trial_params[i]['qB'] for i in range(len(trial_params))])
    seqAB = np.array([trial_params[i]['seqAB']for i in range(len(trial_params))])

    return x,trial_params,model_state,choice12,choiceAB,choiceLR,qAs,qBs,seqAB

if __name__ == '__main__':
    main()

