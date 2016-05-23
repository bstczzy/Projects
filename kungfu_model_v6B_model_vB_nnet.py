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


d_col_drops=['id','relevance','search_term_score', 'product_title_score']
#d_col_drops=['id','relevance']
X_train2 = X_train.drop(d_col_drops,axis=1).values
X_test2 = X_test.drop(d_col_drops,axis=1).values


# Keras Model
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split


#List of activation function:
#softmax,softplus,relu,tanh,sigmoid,hard_sigmoid,linear
#activations=['softmax','softplus','relu','tanh','sigmoid','hard_sigmoid','linear']
activations1=['linear']
activations2=['relu']
layer_dims1=[500,1000,2000]
layer_dims2=[10,100,1000]
X_train3, X_valid3, y_train3, y_valid3 = train_test_split(X_train2, y_train, test_size=0.25, random_state=2009)
activations_res={}

for ai1 in range(len(activations1)):
    for ai2 in range(len(activations2)):
        for dimi1 in range(len(layer_dims1)):
            for dimi2 in range(len(layer_dims2)):
                nnid = activations1[ai1] + ' ' + activations2[ai2] + ' ' + str(layer_dims1[dimi1]) + ' ' + str(layer_dims2[dimi2])
                activations_res[nnid] = []

for ai1 in range(len(activations1)):
    for ai2 in range(len(activations2)):
        for dimi1 in range(len(layer_dims1)):
            for dimi2 in range(len(layer_dims2)):
                model = Sequential()
                model.add(Dense(output_dim=layer_dims1[dimi1], input_dim=X_train2.shape[1]))
                model.add(Activation(activations1[ai1]))
                model.add(Dense(output_dim=layer_dims2[dimi2], input_dim=layer_dims1[dimi1]))
                model.add(Activation(activations2[ai2]))
                model.add(Dense(output_dim=1, input_dim=layer_dims2[dimi2]))
                model.add(Activation('linear'))
                model.compile(loss='mean_squared_error', optimizer='rmsprop')

                model.fit(X_train3, y_train3, nb_epoch=3, batch_size=16)

                loss_res=model.evaluate(X_valid3, y_valid3, batch_size=16, show_accuracy=False, verbose=1, sample_weight=None)
                print(loss_res)

                y_pred = model.predict(X_valid3, batch_size=32)
                y_pred=[max(1.,min(x,3.)) for x in y_pred]
                mse=mean_squared_error(y_valid3, y_pred)**0.5
                print(mse)
                nnid = activations1[ai1] + ' ' + activations2[ai2] + ' ' + str(layer_dims1[dimi1]) + ' ' + str(layer_dims2[dimi2])
                activations_res[nnid].append(mse)

for key in activations_res:
    print(key,activations_res[key])

min_loss=1
for key in activations_res:
    min_loss=np.min([activations_res[key][0],min_loss])

for key in activations_res:
    if activations_res[key][0]==min_loss:
        print(key,activations_res[key][0])


