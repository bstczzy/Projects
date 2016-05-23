# -*- coding: utf-8 -*-
#This version focus on Feature Engineering
#Level 1 Word matching DONE
#Level 2 Word block matching DONE
#Level 3 Material matching
#Level 4 Number / dimension matching [IN PROGRESS]
#Level 5 Add extra words: eg. 1) R-19 -> add r, 19, r19; 2) 2-Light -> add 2, light, 2light; 3) 6ft -> 6, ft, 6-ft;
#Level 5B After adding, if match, then delete from search query, to avoid duplicates
#Level 6 Attribute Info [IN PROGRESS]
import time
start_time = time.time()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
import _pickle as pickle

#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.porter import *
stemmer = PorterStemmer()
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')
import re

import xgboost as xgb


# Load the data into DataFrames
Dir = '/Users/zhenyuz/Documents/Projects/Kaggle/HomeDepot/data'

### Load
pickle_file = '%s/df_v6B_8.pickle'%(Dir)
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  df_all = save['df_all']
  del save  # hint to help gc free up memory
  print('df_all', df_all.shape)
##########################

df_train = pd.read_csv('%s/train.csv'%(Dir), encoding="ISO-8859-1")
df_test = pd.read_csv('%s/test.csv'%(Dir), encoding="ISO-8859-1")
num_train = df_train.shape[0]

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)


df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values
#y_train = pd.DataFrame(df_train['relevance'].values,columns=['relevance'])
X_train =df_train[:]
X_test = df_test[:]
print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))



########### Model Fitting
xgb_model = xgb.XGBRegressor(silent=True, nthread=-1, gamma=0.000001, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None)
#param_grid = {'n_estimators': [550,600,650], 'learning_rate': [0.048,0.05,0.052],'max_depth': [5,6,7]}
param_grid = {'n_estimators': [1000,1100], 'learning_rate': [0.018,0.019],'max_depth': [8]}


#d_col_drops=['id','relevance','search_term_score', 'product_title_score']
d_col_drops=['id','relevance']
X_train2 = X_train.drop(d_col_drops,axis=1).values
model = grid_search.GridSearchCV(estimator = xgb_model, param_grid = param_grid, n_jobs=-1, cv = 2, verbose = 20, refit=False)
model.fit(X_train2, y_train)

print("Grid Scores:")
print(model.grid_scores_)
print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)


#Extracting scores
scores = [x[1] for x in model.grid_scores_]
scores = np.array(scores).reshape(len(param_grid['xgb_model__n_estimators']), len(param_grid['xgb_model__learning_rate']))



#Training error
#y_train_pred = model.predict(X_train2)
#TMSE=fmean_squared_error(y_train, y_train_pred)
#print(TMSE)

f1=open('%s/Grid_score.txt'%(Dir), 'w+')
f1.write(str(model.grid_scores_))

#Prediction
model = xgb.XGBRegressor(n_estimators=1000,learning_rate=0.019,max_depth=8,silent=False, nthread=-1, gamma=0.000001, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None)
model.fit(X_train2, y_train)


X_test2 = X_test.drop(d_col_drops,axis=1).values
y_pred = model.predict(X_test2)
y_pred=[max(1.,min(x,3.)) for x in y_pred]
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('%s/submission_v6_B_xgboost_same_query_score.csv'%(Dir),index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60),2))








### XGboost CV
#'n_estimators':800,
dtrain = xgb.DMatrix(X_train2, label=y_train, missing=-999.)
param = {'max_depth':6, 'eta':1.,'silent':1}
num_round = 1000

print ('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
model_cv=xgb.cv(param, dtrain, num_round, nfold=2,show_progress=True, show_stdv=False,metrics={'error'}, seed = 2009)


### Custom CV
X_train3, X_valid3, y_train3, y_valid3 = train_test_split(X_train2, y_train, test_size=0.25, random_state=2009)
xgb_model = xgb.XGBRegressor(n_estimators=1100,learning_rate=0.019,max_depth=8,silent=False, nthread=-1, gamma=0.000001, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None)
xgb_model.fit(X_train3, y_train3)

y_pred = xgb_model.predict(X_valid3)
y_pred=[max(1.,min(x,3.)) for x in y_pred]
mse=mean_squared_error(y_valid3, y_pred)**0.5
print(mse)









