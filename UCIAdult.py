# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 22:39:50 2017

@author: mithilesh
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing,tree
import matplotlib.pyplot as plt
#data = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
#                  sep=",",header=F,col.names=c("age", "type_employer", "fnlwgt", "education", 
#                                               "education_num","marital", "occupation", "relationship", "race","sex",
#                                               "capital_gain", "capital_loss", "hr_per_week","country", "income"),

ucidata=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
,header=None
,names=["age", "type_employer", "fnlwgt", "education", 
                                               "education_num","marital", "occupation", "relationship", "race","sex",
                                               "capital_gain", "capital_loss", "hr_per_week","country", "income"]
,index_col=False)

ucidata['income']=ucidata['income'].astype('category')
ucidata['income']=pd.Categorical.from_array(ucidata.income).codes
#cardata.dtypes
ucidata.dtypes
#x=pd.get_dummies(ucidata)
le=preprocessing.LabelEncoder()
#le.fit(ucidata['type_employer'])
#ucidata['type_employer']=le.transform(ucidata['type_employer'])
#ucidata['type_employer'].dtype
ucidata=ucidata.apply(le.fit_transform)
ucidata['income']=ucidata['income'].astype('category')
ucidata['race']

ucidata.ix[:,14:14]

ucidata.ix[:,14:15].shape

dt_model= tree.DecisionTreeClassifier()
dt_model=dt_model.fit(ucidata.ix[:,0:14],ucidata.ix[:,14:15])
dt_model.feature_importances_
x=dt_model.predict(ucidata.ix[:,0:14])
np.sum([x==1])
##start LM function
def linear_model_main(X_parameters,Y_parameters,predict_value):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions
##end LM function