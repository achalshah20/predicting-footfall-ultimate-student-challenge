
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
import time, datetime

# reg
tr = pd.read_csv("featuresets/tr2.csv")
ts = pd.read_csv("featuresets/ts2.csv")


def get_train_test(tr,ts):
    y = tr["Footfall"]
    x_train = tr.drop(["Footfall","Unnamed: 0"],1)
    x_test = ts.drop("Unnamed: 0",1)
    return(x_train,y,x_test)

x_train , y ,x_test = get_train_test(tr,ts)

# In[13]:

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV


# In[14]:

params_gbm = [{'min_samples_split':list(range(100,1000,100)),'max_depth' : list(range(3,10,1))}]


# In[15]:

gsearch = GridSearchCV(estimator= GradientBoostingRegressor(n_estimators=250,learning_rate=0.1,=4,max_features="sqrt",subsample=0.8,random_state=10),
                       param_grid = params_gbm, scoring='mean_squared_error',n_jobs=4,cv=5,verbose=10)


# In[ ]:

gsearch.fit(x_train,y)

print("Best parameters set found on development set:")
print()
print(gsearch.best_params_)
print()
print("Grid scores on development set:")
print(gsearch.grid_scores_)
