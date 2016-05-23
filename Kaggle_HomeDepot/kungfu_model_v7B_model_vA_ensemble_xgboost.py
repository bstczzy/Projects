# -*- coding: utf-8 -*-
import time
start_time = time.time()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor

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


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

# Load the data into DataFrames
Dir = '/Users/zhenyuz/Documents/Projects/Kaggle/HomeDepot/data'


df_train = pd.read_csv('%s/train.csv'%(Dir), encoding="ISO-8859-1")
df_test = pd.read_csv('%s/test.csv'%(Dir), encoding="ISO-8859-1")
num_train = df_train.shape[0]


########################   XGBOOST  ########################
### Testing Set Validation
def xgboost_test(df_all_file,d_col_drops,n_estimators,learning_rate,max_depth):
    ### Load
    pickle_file = '%s/%s'%(Dir,df_all_file)
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      df_all = save['df_all']
      del save  # hint to help gc free up memory
      print('df_all', df_all.shape)
    ##########################

    df_train = df_all.iloc[:num_train]
    df_test = df_all.iloc[num_train:]
    id_test = df_test['id']
    y_train = df_train['relevance'].values
    #y_train = pd.DataFrame(df_train['relevance'].values,columns=['relevance'])
    X_train =df_train[:]
    X_test = df_test[:]
    print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))
    X_train2 = X_train.drop(d_col_drops,axis=1).values

    ### Custom CV
    from sklearn.cross_validation import train_test_split
    X_train3, X_valid3, y_train3, y_valid3 = train_test_split(X_train2, y_train, test_size=0.2, random_state=2009)
    xgb_model = xgb.XGBRegressor(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,seed=2016)
    xgb_model.fit(X_train3, y_train3)

    y_pred = xgb_model.predict(X_valid3)
    y_pred=[max(1.,min(x,3.)) for x in y_pred]

    return xgb_model,y_pred,y_valid3

### Predication 
def xgboost_pred(df_all_file,d_col_drops,n_estimators,learning_rate,max_depth):
    ### Load
    pickle_file = '%s/%s'%(Dir,df_all_file)
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      df_all = save['df_all']
      del save  # hint to help gc free up memory
      print('df_all', df_all.shape)
    ##########################

    df_train = df_all.iloc[:num_train]
    df_test = df_all.iloc[num_train:]
    id_test = df_test['id']
    y_train = df_train['relevance'].values
    #y_train = pd.DataFrame(df_train['relevance'].values,columns=['relevance'])
    X_train =df_train[:]
    X_test = df_test[:]
    print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))
    X_train2 = X_train.drop(d_col_drops,axis=1).values

    # Prediction
    xgb_model = xgb.XGBRegressor(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,seed=2016,silent=False, nthread=-1, gamma=0.000001, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, missing=None)
    xgb_model.fit(X_train2, y_train)

    X_test2 = X_test.drop(d_col_drops, axis=1).values
    y_pred = xgb_model.predict(X_test2)
    #y_pred = [max(1., min(x, 3.)) for x in y_pred]
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('%s/submission_v6_B_xgboost_same_query_score.csv' % (Dir),
                                                              index=False)
    print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
    return xgb_model,y_pred



### Testing Set Validation for Random Forests and Extra Trees
def method_test(model,df_all_file,d_col_drops):
    ### Load
    pickle_file = '%s/%s'%(Dir,df_all_file)
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      df_all = save['df_all']
      del save  # hint to help gc free up memory
      print('df_all', df_all.shape)
    ##########################

    df_train = df_all.iloc[:num_train]
    df_test = df_all.iloc[num_train:]
    id_test = df_test['id']
    y_train = df_train['relevance'].values
    #y_train = pd.DataFrame(df_train['relevance'].values,columns=['relevance'])
    X_train =df_train[:]
    X_test = df_test[:]
    print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))
    X_train2 = X_train.drop(d_col_drops,axis=1).values

    ### Custom CV
    from sklearn.cross_validation import train_test_split
    X_train3, X_valid3, y_train3, y_valid3 = train_test_split(X_train2, y_train, test_size=0.2, random_state=2009)
    #xgb_model = xgb.XGBRegressor(n_estimators=n_estimators,max_depth=max_depth,seed=2016)
    model.fit(X_train3, y_train3)

    y_pred = model.predict(X_valid3)
    y_pred=[max(1.,min(x,3.)) for x in y_pred]

    return model,y_pred,y_valid3



### Prediction for Random Forests and Extra Trees
def method_pred(model,df_all_file,d_col_drops):
    ### Load
    pickle_file = '%s/%s'%(Dir,df_all_file)
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      df_all = save['df_all']
      del save  # hint to help gc free up memory
      print('df_all', df_all.shape)
    ##########################

    df_train = df_all.iloc[:num_train]
    df_test = df_all.iloc[num_train:]
    id_test = df_test['id']
    y_train = df_train['relevance'].values
    #y_train = pd.DataFrame(df_train['relevance'].values,columns=['relevance'])
    X_train =df_train[:]
    X_test = df_test[:]
    print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))
    X_train2 = X_train.drop(d_col_drops,axis=1).values
    # Prediction
    model.fit(X_train2, y_train)
    X_test2 = X_test.drop(d_col_drops, axis=1).values
    y_pred = model.predict(X_test2)

    return model, y_pred




def pickle_save(filename,model,y_pred,y_valid):
    ### SAVE Processed Data
    pickle_file = '%s/%s'%(Dir,filename)
    try:
      f = open(pickle_file, 'wb')
      save = {
        'model': model,
        'y_pred': y_pred,
        'y_valild': y_valid,
        }
      #pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      pickle.dump(save, f)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise
    ### SAVED
    return

######   XGBOOST Model 1 ######
df_all_file='df_v6B_8.pickle'
d_col_drops=['id','relevance','search_term_score', 'product_title_score']
n_estimators=1075
learning_rate=0.0185
max_depth=8
xgb_model1, y_test_xg1, y_valid1=xgboost_test(df_all_file, d_col_drops, n_estimators, max_depth)

mse1=mean_squared_error(y_valid1, y_test_xg1)**0.5

pickle_save(filename='xgboost_model_1_testing.pickle',model=xgb_model1,y_pred=ypred1,y_valid=y_valid2)

### Random Forests
model=RandomForestRegressor(max_features=100,max_depth=20,n_estimators = 2000, n_jobs = -1, random_state = 2009, verbose = 1)
model_rf,y_test_rf,y_valid3=method_test(model,df_all_file,d_col_drops)

pickle_save(filename='random_forests_testing.pickle',model=model_rf,y_pred=y_test_rf,y_valid=y_valid3)

mse_rf=mean_squared_error(y_valid3, y_test_rf)**0.5
print(mse_rf)
#0.453313370225

### ETrees

model=ExtraTreesRegressor(max_features=100,max_depth=20,n_estimators = 2000, n_jobs = -1, random_state = 2009, verbose = 1)
model_etree,y_test_etree,y_valid4=method_test(model,df_all_file,d_col_drops)

mse_etree=mean_squared_error(y_valid4, y_test_etree)**0.5
print(mse_etree)
#0.452989264115

######   XGBOOST Model 2 ######
df_all_file='df_v5A_6F.pickle'
d_col_drops=['id','relevance']
n_estimators=1100
learning_rate=0.019
max_depth=8
xgb_model2, y_test_xg2, y_valid2=xgboost_test(df_all_file, d_col_drops, n_estimators, max_depth)

mse2=mean_squared_error(y_valid2, y_test_xg2)**0.5
print(mse2)
#mse1: 0.448681063181
#mse2: 0.447833504104

pickle_save(filename='xgboost_model_2_testing.pickle',model=xgb_model2,y_pred=ypred2,y_valid=y_valid2)

pickle_file = '%s/xgboost_model_2_testing.pickle'%(Dir)
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  xgb_model2 = save['model']
  ypred2=save['y_pred']
  y_valid2=save['y_valild']
  del save  # hint to help gc free up memory


ypred2 = xgb_model2.predict(X_valid3)
ypred2 = [max(1., min(x, 3.)) for x in ypred2]





### Mix Models
mses=[]
#Model ypred_final = w1*ypred1 + w2*ypred2
#w1+w2=1
#w2=r*w1
w1=0.5
w2s=[x/10. for x in range(1,11)]
w3s=[x/10. for x in range(1,11)]
w4s=[x/10. for x in range(1,11)]

y_test_xg1=np.array(y_test_xg1)
y_test_xg2=np.array(y_test_xg2)
y_test_rf=np.array(y_test_rf)
y_test_etree=np.array(y_test_etree)

df=pd.DataFrame(columns=['w1','w2','w3','w4','mse'])
i=0
for w2 in w2s:
    for w3 in w3s:
        for w4 in w4s:
            w1a =w1/(sum([w1,w2,w3,w4]))
            w2a = w2 / (sum([w1,w2,w3,w4]))
            w3a = w3 / (sum([w1,w2,w3,w4]))
            w4a = w4 / (sum([w1, w2, w3, w4]))
            ypred=w1a*y_test_xg1+w2a*y_test_xg2+w3a*y_test_rf+w4a*y_test_etree
            mse=mean_squared_error(y_valid2, ypred) ** 0.5
            mses.append(mse)
            df.loc[i,] = [w1a,w2a,w3a,w4a,mse]
            i+=1


df=df.sort_values(by=['mse'], ascending=[1]).reset_index(drop=True)
df.head()


###### Predication
######   XGBOOST Model 1 ######
df_all_file='df_v6B_8.pickle'
d_col_drops=['id','relevance','search_term_score', 'product_title_score']
n_estimators=1075
learning_rate=0.0185
max_depth=8
xgb_model1a, ypred1a=xgboost_pred(df_all_file,d_col_drops,n_estimators,learning_rate,max_depth)

pd.DataFrame({"id": id_test, "relevance": ypred1a}).to_csv('%s/submission_v7_xgboost1.csv'%(Dir),index=False)


### Random Forests
model=RandomForestRegressor(max_features=100,max_depth=20,n_estimators = 2000, n_jobs = -1, random_state = 2009, verbose = 1)
model_rf_full,y_pred_rf=method_pred(model,df_all_file,d_col_drops)
pd.DataFrame({"id": id_test, "relevance":y_pred_rf}).to_csv('%s/submission_v7_rf.csv'%(Dir),index=False)


### ETrees
model=ExtraTreesRegressor(max_features=100,max_depth=20,n_estimators = 1000, n_jobs = -1, random_state = 2009, verbose = 1)
model_etree_full,y_pred_etree=method_pred(model,df_all_file,d_col_drops)
pd.DataFrame({"id": id_test, "relevance":y_pred_etree}).to_csv('%s/submission_v7_etree.csv'%(Dir),index=False)



######   XGBOOST Model 2 ######
df_all_file='df_v5A_6F.pickle'
d_col_drops=['id','relevance']
n_estimators=1100
learning_rate=0.019
max_depth=8
xgb_model2a, ypred2a=xgboost_pred(df_all_file,d_col_drops,n_estimators,learning_rate,max_depth)

pd.DataFrame({"id": id_test, "relevance": ypred2a}).to_csv('%s/submission_v7_xgboost2.csv'%(Dir),index=False)



### Fit weight: array([0.3846153846153846, 0.3846153846153846, 0.038461538461538464,0.1923076923076923], dtype=object)

id_test = df_test['id']
ypred1a=np.array(ypred1a)
ypred2a=np.array(ypred2a)
y_pred_rf=np.array(y_pred_rf)
y_pred_etree=np.array(y_pred_etree)

ws=df.loc[0,].values[:4]
y_pred=ws[0]*ypred1a+ws[1]*ypred2a+ws[2]*y_pred_rf+ws[3]*y_pred_etree
y_pred=[max(1.,min(x,3.)) for x in y_pred]
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('%s/submission_v7_mix4a.csv'%(Dir),index=False)


#Weight 2: w1=5.
id_test = df_test['id']
ypred1a=np.array(ypred1a)
ypred2a=np.array(ypred2a)
y_pred_rf=np.array(y_pred_rf)
y_pred_etree=np.array(y_pred_etree)

ws=df.loc[0,].values[:4]
y_pred=ws[0]*ypred1a+ws[1]*ypred2a+ws[2]*y_pred_rf+ws[3]*y_pred_etree
y_pred=[max(1.,min(x,3.)) for x in y_pred]
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('%s/submission_v7_mix4b.csv'%(Dir),index=False)


#Weight 2: w1=0.5,ws=[0.25, 0.5, 0.05, 0.2]
id_test = df_test['id']
ypred1a=np.array(ypred1a)
ypred2a=np.array(ypred2a)
y_pred_rf=np.array(y_pred_rf)
y_pred_etree=np.array(y_pred_etree)

ws=df.loc[0,].values[:4]
y_pred=ws[0]*ypred1a+ws[1]*ypred2a+ws[2]*y_pred_rf+ws[3]*y_pred_etree
y_pred=[max(1.,min(x,3.)) for x in y_pred]
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('%s/submission_v7_mix4c.csv'%(Dir),index=False)





### SAVE Processed Data
pickle_file='%s/df_v7_test_pred_mix4.pickle'%(Dir)
try:
    f = open(pickle_file, 'wb')
    save = {
        'y_valid2':y_valid2,
        'y_test_xg1': y_test_xg1,
        'y_test_xg2': y_test_xg2,
        'y_test_rf': y_test_rf,
        'y_test_etree':y_test_etree,
        'ypred1a':ypred1a,
        'ypred2a':ypred2a,
        'y_pred_rf':y_pred_rf,
        'y_pred_etree':y_pred_etree
    }
    # pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(save, f)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
### SAVED



