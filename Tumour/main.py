# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:00:15 2020

@author: Aakash Babu
"""
import pandas as pd

def get_input():
    train = pd.read_csv("Train.csv")
    test = pd.read_csv("Test.csv")
    return train,test

train,test = get_input()

#--------------Printing for the null values in the train ---------#

def null_value():
    print(train.isnull().sum())
    print(test.isnull().sum())

#   null_value()

# ------- So there is no null values so we can proceed to the next steps

'''
In this problem it is nothing but the Supervised(Regression) problem 

where the target variable in tumor_size 

'''
# ------ let us find what are the qualitative and quanditative values

def print_unique():
    for val in train.columns:
        print("The column {} is {}".format(val,len(train[val].unique())))
        
#  print_unique()

# ------ from this we come to know that these features are quanditative ---------#

'''
Let us discuss the features and uses

mass_npea:  the mass of the area understudy for melanoma tumor
size_npear: the size of the area understudy for melanoma tumor
malign_ratio: ration of normal to malign surface understudy
damage_size: unrecoverable area of skin damaged by the tumor
exposed_area: total area exposed to the tumor
std_dev_malign: standard deviation of malign skin measurements
err_malign: error in malign skin measurements
malign_penalty: penalty applied due to measurement error in the lab
damage_ratio: the ratio of damage to total spread on the skin
tumor_size: size of melanoma_tumor

'''

# This problem doesn't include good feature engineering since it doesn't have any null values

X = train.drop(['tumor_size'],axis=1)
y = train['tumor_size']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=7,test_size=0.33)

'''
# Decision tree Regressor for regression

from sklearn.tree import ExtraTreeRegressor

cls = ExtraTreeRegressor()
cls.fit(X_train,y_train)

pred = cls.predict(X_test)

from sklearn.metrics import mean_squared_error

print("The mean squared error is {}".format(mean_squared_error(pred,y_test)))

# output The mean squared error is 34.18461074039417

import seaborn as sns

sns.distplot(y_test-pred)


RandomForestRegressor()
The mean squared error is 16.776179302803882

After hypter tuning
RandomForestRegressor(max_depth=30, max_features='log2', n_estimators=150)

The mean squared error is 16.553698616492934

ExtraTreesRegressor

The mean squared error is 15.65352146904475
'''

# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor



#cls = ExtraTreesRegressor(n_estimators=150,max_depth=30,random_state=0)
#cls = CatBoostRegressor(verbose=False)
cls = ExtraTreesRegressor()
cls.fit(X_train,y_train)

pred = cls.predict(X_test)
from sklearn.metrics import mean_squared_error,SCORERS

from sklearn.model_selection import cross_val_score
print(cross_val_score(cls,X,y,cv=5,scoring="neg_mean_squared_error"))
#print(SCORERS.keys())

print("The mean squared error of {} is {}".format(cls,mean_squared_error(pred,y_test)))


'''
# Hypertuning the Random Forest Regressor

from sklearn.model_selection import RandomizedSearchCV

params={
        "n_estimators":[50,100,150,200],
        "criterion":['mse','mae'],
        "max_depth":[10,20,30],
        "max_features":["auto", "sqrt", "log2"]        
        }

random = RandomizedSearchCV(cls,params,random_state=0,n_jobs=-1,verbose=1)
random.fit(X_train,y_train)

print(random.best_params_)
print(random.best_estimator_)
'''

# Convert the submission file for submission
sub = cls.predict(test)
submission = pd.DataFrame()
submission['tumor_size'] = sub
submission.to_csv("Submission-extratree.csv",index=False)
