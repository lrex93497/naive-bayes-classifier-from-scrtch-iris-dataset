## complete date: 26/9/2022@11:51:33 (GMT+8)

class NBC:
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes
        print("initialised")
        print("########################################")

    def fit(self, Xtrain, ytrain):
        
        ##caculate prior probility (class probabilities in question) as self.log_prior_p in log10, prior_p as a original number
        lenght_y = len(ytrain)
        self.lenght_y = lenght_y
        unique_elements, counts_elements = np.unique(y, return_counts=True)
        ##unique_elements = [0,1,2] in iris case
        self.prior_p = counts_elements/lenght_y
        self.log_prior_p = np.log10(counts_elements/lenght_y)
        

        ## split Xtrain acording to category in to X_category[0 to n] 3 in iris
        category_count = self.num_classes
        X_category = [None] * self.num_classes
        
        for i in range(category_count):
            temp=[]
            for v in range(lenght_y):
                if y[v] == i:
                    temp.append(X[v])
                   
            X_category[i] = temp
        
        X_category = np.array(X_category)   


        ##caculate condi distribution as conditionally distributions_std and self.condi_distribution_mean
        self.condi_distribution_std = [None] * self.num_classes
        self.condi_distribution_mean = [None] * self.num_classes

        for i in range(self.num_classes):
            self.condi_distribution_std[i] = np.std(X_category[i], axis=0)
        for i in range(self.num_classes):
            self.condi_distribution_mean[i] = np.mean(X_category[i], axis=0)

        self.condi_distribution_std = np.array(self.condi_distribution_std)
        self.condi_distribution_mean = np.array(self.condi_distribution_mean)
        ## if std = 0, make it become 0.000001 to avoid devide by 0
        self.condi_distribution_std[self.condi_distribution_std == 0] = 0.000001

        ##combine as conditionally distributions self.condi_distribution 
        # [[[mean, std], ....,[mean, std]],...,[[std, mean], ....,[std, mean]]]
        self.condi_distribution = np.dstack((self.condi_distribution_mean, self.condi_distribution_std))

        print("########################################")
        print("fit complete:\n prior probability (class probailities) ([0,1,2]) = ") 
        print(self.prior_p )
        print("fit complete:\n prior probability (class probailities) ([0,1,2]) in log10= ") 
        print(self.log_prior_p) 
        print("\nconditional distributions ([[[mean, std],...,[mean, std]],...,[[std, mean],...,[std, mean]]]):")
        print(self.condi_distribution)
        return 

    def predict(self, Xtest):
        lenght_Xtest = len(Xtest)
        feature_no = len(self.feature_types)
        class_no = self.num_classes

        log_conditional_p = []
        
        ## found log10 conditional probability as log_conditional_p 
        for i in range(lenght_Xtest):
            temp = []
            for u in range(class_no):
                temp2 = 0.0
                for v in range(feature_no):
                    if v==0:
                        temp2 = math.log(scipy.stats.norm(self.condi_distribution[u][v][0], self.condi_distribution[u][v][1]).pdf(Xtest[i][v]), 10)
                    temp2 = temp2 + math.log(scipy.stats.norm(self.condi_distribution[u][v][0], self.condi_distribution[u][v][1]).pdf(Xtest[i][v]), 10)
                temp.append(temp2)
            log_conditional_p.append(temp)
        
        log_conditional_p = np.array(log_conditional_p)
        prediction_p = self.log_prior_p + log_conditional_p
        yhat = np.argmax(prediction_p, axis=1)

        print("########################################")
        print("prediction complete:")
        return yhat

    

from symbol import test_nocond
import numpy as np
import scipy.stats
import math
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris['data'], iris['target']

N, D = X.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]


nbc = NBC(feature_types=['r','r','r','r',], num_classes=3)
nbc.fit(Xtrain, ytrain)
yhat = nbc.predict(Xtest)

print("prediction result:")
print(yhat)
test_accuracy = np.mean(yhat == ytest)
print("\nPrediction accuracy:")
print(test_accuracy)
