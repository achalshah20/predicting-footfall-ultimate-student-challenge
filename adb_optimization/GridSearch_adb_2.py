
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os

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

from sklearn.ensemble import AdaBoostRegressor
from sklearn.grid_search import GridSearchCV


# ### ADB

# In[14]:

params_adb = [{'learning_rate' : [1,1.2,1.5,1.7,2] ,'n_estimators' : [300,400,500]}]


# In[15]:

gsearch = GridSearchCV(estimator= AdaBoostRegressor(),
                       param_grid = params_adb, scoring='mean_squared_error',n_jobs=50,cv=5,verbose=10)


# In[ ]:

gsearch.fit(x_train,y)


print(gsearch.best_params_)
print(gsearch.grid_scores_)
