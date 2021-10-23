ReadMe:
#######################################################
B2M.py
This file defines the class of B2M.
Here is an example of using B2M for pool based active learning:


#Create the index that partition the dataset into training set, test test and candidate pool set. X: input matrix. Y:multi-label matrix.
train_index,candidate_index,test_index=split(X,label=Y,train_rate=0.01,candidate_rate=0.4,test_rate=0.5,seed=2333,even=True)
#The number of samples selected for each active learning iteration
batch=1
#result list used to store the test performance(AUC score in our case) for each active learning iteration
res=[]
for i in range(500):
    print('iter%d'%(i))
	#Create the object of B2M model.
    model1=B2M()
	#Train the model using training data. K:is the maximum number of components.
	#k_threds: the parameter rho that used to control the number of components preserved for prediction.
    model1.fit(X[train_index,:],Y[train_index,:],K=10,kernel_function=RBF(length_scale=1),K_threds=0.75)
	#Learn iter:specifies how many iterations we want to run to update latent variables. Can be replaced with other convergence criteria such as change of ELBO/latent variable errors.
    model1.learn(learnIter=15)
	#Predict E[y] over the test dataset. This step is used to monitor the active learning behavior, and is not a necessary step for real-world active learning applications.
	#res='full' will provide more detailed predictions including the learned components and component assignments for each test data instance.
	#useAllK: if True, will use all K components for prediction. Else, will use components selected by K_threds to predict.
    proba1=model1.predict_proba(X[test_index,:],res='simple',mode='ntest',useAllK='True')
	#active sampling. 
	#eta: the parameter used to control the predicted label covariance and variance. sampleType: when set to 'en', this correspond to the proposed sampling function. 
    sampleRes=model1.ALsample(X[candidate_index,:],eta=0,test=False,sampleType='en',testX=None,testY=None)
    #update the current training set and move on.
	candInd=sampleRes[0:batch]
    batchList=[candidate_index[i] for i in candInd]
    train_index=train_index+batchList
    candidate_index=[x for x in candidate_index if x not in batchList ]
    AUC=prediction_scores(Y[test_index],proba1,output='all')
    print(AUC)
    res.append(AUC[0])
	
	