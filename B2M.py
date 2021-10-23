# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.gaussian_process.kernels import RBF
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
import os
from B2M_util import *

#determine k, the number of component automatically.

def syntheticDataGen(xLeft,xRight,yUp,yDown,nPerRow,sigma2=1.5,samplePerClass=100,thres=0.25,dim=2,additionalRule=False):
    #centorids of the Gaussian clusters are arranged in the grid range(xLeft,xRight,nPerRow),range(yLeft,yRight,nPerRow)
    samples=[]
    pdfs=[]
    clusterIndex=[]#used for manually labeling a specific cluster
    nClusters=0
    restDims=[0 for i in range(dim-2)]#all centers of Gaussian in the same 2D plane. So the restDims are dim-2
    for yInd,centerY in enumerate(range(yUp,yDown,-nPerRow)):
        for xInd,centerX in enumerate(range(xLeft,xRight,nPerRow)):
            nClusters+=1
            a=np.random.multivariate_normal([centerX,centerY]+restDims, np.eye(dim)*sigma2, samplePerClass)
            n=multivariate_normal(mean=[centerX,centerY]+restDims, cov = np.eye(dim)*sigma2)
            samples.append(a)
            pdfs.append(n)
            clusterIndex+=[(xInd,yInd) for i in range(samplePerClass)]
    x=np.concatenate(samples,axis=0)            
    y=np.zeros([samplePerClass*nClusters,int(0.5*nClusters)])
    trueProb=np.zeros([samplePerClass*nClusters,nClusters]) 
    print('nclusters:%d'%(nClusters))
    for i in range(nClusters):
        if(i<nClusters*0.5):
            y[i*samplePerClass:(i+1)*samplePerClass,i]=1    
        else:#repeat the labeling
            y[i*samplePerClass:(i+1)*samplePerClass,int(i-nClusters*0.5)]=1   
        trueProb[:,i]=pdfs[i].pdf(x)
    trueProb=normalize(trueProb,norm='l1') 
    for i in range(x.shape[0]):
        for j in range(len(pdfs)):
            if(trueProb[i,j]>thres):
                if(j<int(0.5*nClusters)):
                    y[i,j]=1
                else:
                    y[i,int(j-nClusters*0.5)]=1
    print(y.shape)
    def labelCluster(clusterList,labelList):
        return
    if(additionalRule):
        #randomly pick 10% data to generate 2 new labels
        cardi=np.sum(y,axis=1)
        #add labels indicating cardi=2 and cardi=4
        y=np.concatenate([y,np.zeros([y.shape[0],2])],axis=1)
        for i in range(y.shape[0]):
            if(cardi[i]==3):
                y[i,-2]=1
            elif(cardi[i]>3):
                y[i,-1]=1
        #add two logical labels wrt l1 l2, l2 l3,  labels(and or opertation)
        y=np.concatenate([y,np.zeros([y.shape[0],8])],axis=1)
        for i in range(y.shape[0]):
            if(y[i,0] and y[i,1]): #over lapping zone label. 
                y[i,-8]=1
            if(y[i,1] and y[i,2]): # over lapping zone label
                y[i,-7]=1
            if(y[i,2] and y[i,3]): #over lapping zone label. 
                y[i,-6]=1
            if(y[i,4] and y[i,5]): # over lapping zone label
                y[i,-5]=1
        
        tem=y[:,0]
        for i in [5]:
            tem=np.logical_or(tem,y[:,i])#index of data with label 0,5
        index=np.where(tem==1)[0]
        np.random.shuffle(index)
        #the 'contain' relationship
        y[index,-4]=1 #data with label 0 or 5 has this label -5
        y[index[0:int(0.3*len(index))],-3]=1#1/3 of the data with label -5 has label -6
        np.random.shuffle(index)
        y[index[0:int(0.3*len(index))],-2]=1 #1/3 of the data with label -5 has label -7        
        #exclusive relationship
        print('what')
        for i in range(y.shape[0]):
            if(y[i,-2] and not y[i,-3]):
                y[i,-1]=1
    print('label sparsity:%f'%(np.sum(y)/(y.shape[0]*y.shape[1])))
    print('number of instance cardi=2: %f cardi=3 %f cardi=4 %f cardi=5 %f' %(len(np.where(np.sum(y,axis=1)==2)[0]),len(np.where(np.sum(y,axis=1)==3)[0]),len(np.where(np.sum(y,axis=1)==4)[0]),len(np.where(np.sum(y,axis=1)==5)[0])))
    return x,y,pdfs,clusterIndex,trueProb


class B2M():
    def log_BetaNorm(self,a,b):
        def approxLogGamma(x):
            return (x-0.5)*np.log(x)-x+0.5*np.log(2*np.pi)
        return approxLogGamma(a+b)-approxLogGamma(a)-approxLogGamma(b)
    def discretisize_phi(self,phi):
        tem=np.zeros(phi.shape)
        for i in range(phi.shape[0]):
            tem[i,np.argmax(phi[i,:])]=1
        return tem
    
    def sigmoid(self,x):
        #prevent overflow
        if(-x>1e2):
            return  1/(1+np.exp(1e2))
        else:
            return 1/(1+np.exp(-x))
        
    def __init__(self):
        self.record=True    
    
    def fit(self,X,Y,K=25,seed=5,kernel_function=None,K_threds=1):
        self.K_threds=K_threds
        #initialize data/label dimensions.
        self.X=X#copy of data samples N*M
        self.Y=Y#copy of labels N*L
        self.K=K#number of mixture components
        self.N=X.shape[0]#number of samples
        self.M=X.shape[1]#number of attributes
        self.L=Y.shape[1]#number of labels for each data sample
        #initialize variational parameters.
        self.A=np.random.random([self.K,self.L])+1#prior of the Beta distribution
        self.B=np.random.random([self.K,self.L])+1#prior of the Beta distribution
        self.A_hat=self.A.copy()#parameter of the Beta distribution
        self.B_hat=self.B.copy()#parameter of the Beta distribution
        self.kernel_function=kernel_function
        self.Kmask=np.ones(K)
        self.Alpha=np.random.random([self.N,1])+1e-3#parameter of the Gamma distribution
        self.Beta=np.ones([self.N,1])*self.K#parameter of the Gamma distribution. This parameter is fixed during the iterative update process.
        self.C=np.random.random([self.N,self.K])#the second parameter of the PG distribution(the first parameter does not participate in the update.)
        self.M_hat=np.random.random([self.N,self.K])-0.5#parameters of the GPs.minus 0.5 to make M_hat cernter at 0. since the prior M_0=0 we don't have to put it here.
        self.Sigma=np.zeros([self.K,self.N,self.N])
        for i in range(self.K):
            if (kernel_function is None):
                self.Sigma[i,:,:]=np.eye(self.N)#Prior of the GPs K*N*N
            else:
                self.Sigma[i,:,:]=kernel_function(self.X)
        self.Sigma_hat=self.Sigma.copy()#parameters of the GPs
        self.Phi=np.random.random([self.N,self.K]) #parameters of Z(categorical distribution)
        self.Phi=normalize(self.Phi,norm='l1')
        self.Gamma=np.random.random([self.N,self.K])#parameters of Possion.
        self.Sigma_inv=np.linalg.pinv(self.Sigma[0,:,:])#the inverse of the Gram mat. Need to be used multiple times so we make it global here.
    
    
    
    def computeMu(self):
        mu=np.zeros([self.K,self.L])
        for l in range(self.L):
            for k in range(self.K):
                mu[k,l]=self.A_hat[k,l]/(self.A_hat[k,l]+self.B_hat[k,l])
        self.mu=mu
        
    def update(self):
        #Block that update Alpha, C, Gamma, and Phi
        for n in range(self.N):
            gamma_update=np.exp(dga(self.Alpha[n]))/self.Beta[n]#this term is irrelevant to K so can be put here to save some computations.
            alpha_update=1#calculate the term  needed in updating alpha
            for k in range(self.K):
                f_square=np.sqrt(np.abs(np.power(self.M_hat[n,k],2)+self.Sigma_hat[k,n,n]))#the second momentum of f, this term is shared by both Gamma and C update rules.
                if(f_square>1e2):
                    f_square=1e2
                tem_exp=-0.5*self.M_hat[n,k]#prevent overflow
                if(tem_exp>1e2):
                    tem_exp=1e2
                self.Gamma[n,k]= gamma_update * np.exp(tem_exp)/np.cosh(f_square)  #update Gamma eq(98)
                #update C eq(108)
                self.C[n,k]=f_square
                phi_update=1#calculate the term needed in updating phi.
                for l in range(self.L):
                    phi_update=phi_update+self.Y[n,l]*(dga(self.A[k,l])-dga(self.A[k,l]+self.B[k,l]))
                #prevent overflow
                prevent=self.M_hat[n,k]*0.5 + phi_update
                if (prevent>1e2):
                    prevent=1e2
                else:
                    self.Phi[n,k]=np.exp(prevent) #update Phi eq(112)
                alpha_update=alpha_update+self.Gamma[n,k] #update Alpha eq(89)
            self.Alpha[n]=alpha_update #update Alpha eq(89)
        self.Phi[self.Phi>1e5]=1e5#avooid inf phi when K is large.            
        self.Phi[self.Phi<1e-5]=1e5#lower bound of phi. make computation faster
        self.Phi=normalize(self.Phi,norm='l1')#Normalize Phi
        #Block that update A,B
        for k in range(self.K):
            for l in range(self.L):
                temA=0
                temB=0
                for n in range(self.N):
                    temA+=self.Phi[n,k]*self.Y[n,l]
                    temB+=self.Phi[n,k]*(1-self.Y[n,l])
                self.A_hat[k,l]=self.A[k,l]+temA #update A,B eq(85,86)
                self.B_hat[k,l]=self.B[k,l]+temB #update A,B eq(85,86)
        #Block that update M_hat, Sigma_hat
        self.disc_Phi=self.discretisize_phi(self.Phi)#get the discrete version of Phi used as E[z]
        for k in range(self.K):
            diag=np.zeros(self.N)#construct the diagonal
            for n in range(self.N):
                diag[n]=(self.disc_Phi[n,k]+self.Gamma[n,k])/(2*self.C[n,k]) * np.tanh(self.C[n,k]*0.5)#E[omega_{nk}]
                #diag[n]=(self.Phi[n,k]+self.Gamma[n,k])/(2*self.C[n,k])#E[omega_{nk}]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.Sigma_hat[k,:,:]=np.linalg.pinv(self.Sigma_inv+np.diag(diag))#update Sigma_hat eq(103)
            #exception cpature: if the inv gives invalid value, record the problemetic operants. 
            if(np.isnan(self.Sigma_hat[k,:,:]).any()):
                self.errorDiag=diag
                self.errorComponent=k
                return
            self.M_hat[:,k]=0.5*np.dot(self.Sigma_hat[k,:,:],(self.disc_Phi[:,k] - self.Gamma[:,k]))#update M_hat eq(102)
            #self.M_hat[:,k]=0.5*np.dot(self.Sigma_hat[k,:,:],(self.Phi[:,k] - self.Gamma[:,k]))#update M_hat eq(102)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
    def learn(self, learnIter=15):
        for i in range(learnIter):
            if(i%4==0):
                pass#print('Update parameters...Round %d'%(i))
            self.update()
        self.computeMu()
        #get conf of components
        confList=np.sum(self.A_hat+self.B_hat,axis=1)
        confMean=np.mean(confList)#use median
        #filter components
        self.selectedK=np.where(confList>=confMean*self.K_threds)[0]
        print('%d components are selected according to K thres:%f'%(len(self.selectedK),self.K_threds))
    
    def predict_proba(self,test,res='simple',mode='ntest',sample=False,sampleSize=10,useAllK=True):
        NN=test.shape[0]
        m_new=np.zeros([NN,self.K])
        sigmoids=np.zeros([NN,self.K])
        phi_new=np.zeros([NN,self.K])
        prob=np.zeros([NN,self.L])
        kappa=self.kernel_function(self.X,test)#compute kappa
        kappa_new=self.kernel_function(test,test)
        if(useAllK is True):
            k_list=list(range(self.K))
        else:
            k_list=self.selectedK
        if (mode=='test'):
            optPhi,cost=getPseudoTarget(self.mu,self.Y,norm=False) #for testing
            self.optPhi=optPhi
            MOR=MultiOutputRegressor(BayesianRidge(fit_intercept=False)).fit(self.X,self.optPhi) #for testing
            self.phi_predict_model=MOR
            pre_phi=MOR.predict(test)
            proba=np.dot(pre_phi,self.mu)
            if(res=='full'):
                return proba,phi_new,self.mu,kappa.T
            else:
                return MOR.predict(kappa.T)
        elif(sample is False):
            for k in k_list:
                m_new[:,k]=0.5* np.linalg.multi_dot([kappa.T,self.Sigma_hat[k,:,:],self.Sigma_inv,(self.disc_Phi[:,k]-self.Gamma[:,k])])#!!!!!!!!!!!!!!!!!!!!!!!!!!!
                for n in range(NN):
                    sigmoids[n,k]=self.sigmoid(m_new[n,k])
            sigmoid_sum=np.sum(sigmoids,axis=1)    
            
            for n in range(NN):
                phi_new[n,:]=sigmoids[n,:]/sigmoid_sum[n]
            
            for n in range(NN):
                for k in range(self.K):
                    #res[n,:]+=mm[n,k]*self.mu[k,:]
                    prob[n,:]+=phi_new[n,k]*self.mu[k,:]
            self.Kappa=kappa
            self.Phi_new=phi_new#the component coefficient for each test.
            self.m_new=m_new
            self.sigmoids=sigmoids
            if(res=='full'):
                return prob,phi_new,self.mu,kappa.T
            else:
                return MOR.predict(kappa.T)
            
        elif (sample is True):#MC sample
            phi_new=np.zeros([sampleSize,NN,self.K])
            for n in range(NN):
                vec=np.ones([1,test.shape[1]])
                vec[0,:]=test[n,:]
                kappa=self.kernel_function(self.X,vec)#compute kappa
                kappa_new=self.kernel_function(vec,vec)
                for k in k_list:
                    m_new=0.5* np.linalg.multi_dot([kappa.T,self.Sigma_hat[k,:,:],self.Sigma_inv,(self.disc_Phi[:,k]-self.Gamma[:,k])])#!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    sigma_new=np.linalg.multi_dot([kappa.T , self.Sigma_inv , self.Sigma_hat[k,:,:], self.Sigma_inv , kappa , ]) + kappa_new - np.linalg.multi_dot([kappa.T  , self.Sigma_inv , kappa])
                    s=np.random.normal(m_new,sigma_new, sampleSize)
                    for t in range(sampleSize):
                        phi_new[t,n,k]=self.sigmoid(s[t])
            #in sample divide sum sigmoid
            for t in range(sampleSize):
                sumTerm=np.sum(phi_new[t,:,:],axis=1)
                for n in range(NN):
                    phi_new[t,n,:]=phi_new[t,n,:]/sumTerm[n]
            #avg over samples
            phi_new=np.mean(phi_new,axis=0)
            for n in range(NN):
                for k in k_list:
                    #res[n,:]+=mm[n,k]*self.mu[k,:]
                    prob[n,:]+=phi_new[n,k]*self.mu[k,:]
            if(res=='full'):
                return prob,phi_new,self.mu,kappa
            else:
                return prob
                    

    def ALsample(self,candidate,eta=0,test=False,sampleType='en',topk=5,testX=None,testY=None,aucThres=0.55,var_multiplier=1e5):
        #var_multiplier rescale the predicted var of phi.
        NN=candidate.shape[0]
        m_new=np.zeros([NN,self.K])
        sigma_new=np.zeros([self.K,NN,NN])
        kappa=self.kernel_function(self.X,candidate)#compute kappa
        kappa_new=self.kernel_function(candidate,candidate)#compute kappa
        sigmoids=np.zeros([NN,self.K])
        for k in range(self.K):
            m_new[:,k]=0.5* np.linalg.multi_dot([kappa.T,self.Sigma_hat[k,:,:],self.Sigma_inv,(self.disc_Phi[:,k]-self.Gamma[:,k])])#!!!!!!!!!!!!!!!!!!!!!!!!!!!
            sigma_new[k,:,:]=np.linalg.multi_dot([kappa.T , self.Sigma_inv , self.Sigma_hat[k,:,:], self.Sigma_inv , kappa , ]) + kappa_new - np.linalg.multi_dot([kappa.T  , self.Sigma_inv , kappa])
        score=np.zeros(candidate.shape[0])
        if(sampleType=='rand'):
            tem=np.array(range(candidate.shape[0]))
            np.random.shuffle(tem)
            return tem
        elif(sampleType=='var'):
            sampleSize=3
            piVar=np.zeros(candidate.shape[0])
            if(self.kernel_function is not None):
                    self.candGram=self.kernel_function(candidate,self.X)
            else:
                self.candGram=candidate
            self.candGram=np.concatenate([np.ones([self.candGram.shape[0],1]),self.candGram],axis=1)       
            samples=np.zeros([sampleSize,NN,self.K])
            for i in range(sampleSize):
                pi_sample=np.zeros([NN,self.K])
                sumPi=0
                f_sample=np.zeros([NN,self.K])
                for k in range(self.K):
                    f_sample[:,k]=np.random.multivariate_normal(m_new[:,k],sigma_new[k,:,:], 1)
                for n in range(NN):
                    for k in range(self.K):
                        pi_sample[n,k]=self.sigmoid(f_sample[n,k])
                for n in range(NN):
                    pi_sample[n,:]=pi_sample[n,:]/np.sum(pi_sample[n,:])
                samples[i,:,:]=pi_sample
            for n in range(NN):
                mean_var=np.mean(np.var(samples[:,n,:],axis=0))#avraage over k
                piVar[n]=mean_var
            index=np.argsort(piVar)[::-1]
            print('max,min vars are %f,%f'%(max(piVar),min(piVar)))
            if (test is True):
                return[index,piVar]    
            else:
                return index#this is the index in candidate
        elif(sampleType=='max_card'):
            #index=np.argmax(score)#detailed res:[logDetCov,piVar]
            index=np.argsort(np.sum(testY,axis=1))[::-1]
            return index
        elif(sampleType=='KL'):#maximize the training component assignment(avg) and predicted component distance
            proba1,phi2,mu1,kappa=self.predict_proba(candidate,res='full',mode='ntest')
            phiMean=np.mean(self.Phi,axis=0)
            score=np.sqrt(np.sum(np.power(phi2-phiMean,2),axis=1))
            index=np.argsort(score)[::-1]
            return index
        else:
            t1=time.time()
            Ey,Pi,Mu,kap=self.predict_proba(candidate,res='full')
            t2=time.time()
            print('predict over pool takes %f sec'%(t2-t1))
            #Label covariance
            logDetCov=np.zeros(candidate.shape[0])
            cov=[]
            if(sampleType=='en_topk_auc'):
                subInd=self.subCovInd(testX,testY,aucThres=aucThres)
                cov=np.zeros([candidate.shape[0],len(subInd),len(subInd)])
                print('The reduced size of cov would be %f bt %f'%(len(subInd),len(subInd)))
            else:
                cov=np.zeros([candidate.shape[0],self.L,self.L])
            for n in range(candidate.shape[0]):
                t3=time.time()
                temRes=np.zeros([self.L,self.L])
                for k in range(self.K):
                    #Lambda[k,:,:]=np.diag(Mu[k,:]*(1-Mu[k,:]))
                    Lambda_k=np.diag(Mu[k,:]*(1-Mu[k,:]))
                    temRes+=Pi[n,k]*(Lambda_k+np.outer(Mu[k,:],Mu[k,:]))
                cov_n=temRes-np.outer(Ey[n,:],Ey[n,:])   
                if(sampleType=='en'):
                    cov[n,:,:]=cov_n
                    logDetCov[n]=np.log(np.linalg.det(cov_n))
                elif(sampleType=='genCor'):
                    cov[n,:,:]=cov_n
                    logDetCov[n]=np.sqrt(1-np.linalg.det(cov_n)/np.prod(np.diagonal(cov_n) ))
                elif(sampleType=='en_topk_auc'):
                    subCov=cov_n[subInd,:]
                    subCov=subCov[:,subInd]
                    logDetCov[n]=np.log(np.linalg.det(subCov))
                    cov[n,:,:]=subCov
                elif(sampleType=='eigen'):#use product of topK eigen vals to approximate det(cov)
                    w,v=np.linalg.eigh(cov_n)
                    w=w[-topk:]
                    logDetCov[n]=np.prod(w)
                t4=time.time()
                if (n==0):
                    print('compute en for one sample takes %f sec'%(t4-t3))
            #var of pi(    signment)
            sampleSize=5
            piVar=np.zeros(NN)
            if (eta!=0):
                if(self.kernel_function is not None):
                    self.candGram=self.kernel_function(candidate,self.X)
                else:
                    self.candGram=candidate
                #sample:phi_nk=sigmoid(f_nk)/sum_n(sigmoid(f_nk))
                samples=np.zeros([sampleSize,NN,self.K])
                for i in range(sampleSize):
                    pi_sample=np.zeros([NN,self.K])
                    sumPi=0
                    f_sample=np.zeros([NN,self.K])
                    for k in range(self.K):
                        f_sample[:,k]=np.random.multivariate_normal(m_new[:,k],sigma_new[k,:,:], 1)
                    for n in range(NN):
                        for k in range(self.K):
                            pi_sample[n,k]=self.sigmoid(f_sample[n,k])
                    for n in range(NN):
                        pi_sample[n,:]=pi_sample[n,:]/np.sum(pi_sample[n,:])
                    samples[i,:,:]=pi_sample
                for n in range(NN):
                    mean_var=np.mean(np.var(samples[:,n,:],axis=0))#avraage over k
                    piVar[n]=mean_var
            score=logDetCov+eta*piVar
            #index=np.argmax(score)#detailed res:[logDetCov,piVar]
            index=np.argsort(score)[::-1]
            if (test is True):
                return[index,logDetCov,cov,piVar]    
            else:
                return index#this is the index in candidate
  
    
    
def checkSparse(Y):
    print(np.sum(Y)/Y.shape[0]/Y.shape[1])
                
                
def prediction_scores(true,pred,output='micro_auc',rare_labels=-1):
    #the true is n*l mat where n is the num of test l is the num of label
    if(output=='micro_auc'):
        return roc_auc_score(true,pred,average='micro')
    else:
        if(rare_labels>0):
            rareInd=np.argsort(np.sum(true,axis=0))#sort by labelFreq
            rareInd=rareInd[0:rare_labels]
            true=true[:,rareInd]
            pred=pred[:,rareInd]
        miAuc=roc_auc_score(true,pred,average='micro')
        maAuc=roc_auc_score(true,pred,average='macro')
        covErr=coverage_error(true, pred)#on average, each row of prediction requires how many positive predictions to predict all true positive labels.(row wised measurement)
        LRAP= label_ranking_average_precision_score(true, pred)#label ranking avg precision for each ground truth label, what fraction of higher-ranked labels were true labels?(col wised measurement)
        weighted=roc_auc_score(true,pred,average='weighted')
        return [miAuc,maAuc,covErr,LRAP,weighted]        
            
            

    
        
        
        
        