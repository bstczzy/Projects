""" Random Forest and Boosting
Author : Zhenyu Zhao
""" 
import pandas as pd
import numpy as np
import csv as csv

Dir='/Users/zhenyuz/Documents/Projects/Kaggle/Titanic/data'

import os
os.chdir(Dir)


# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

wordcount=dict()
names=train_df.Name.values
for i in range(len(names)):
    for word in names[i].split():
        if word not in wordcount:
            wordcount[word]=1
        else:
            wordcount[word]+=1

for word in wordcount.keys():
    if wordcount[word]>50:
        print word, wordcount[word]




import operator
sorted_word = sorted(wordcount.items(), key=operator.itemgetter(1))



# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.loc[(train_df.Embarked.isnull()), 'Embarked' ] = train_df.Embarked.dropna().mode().values

#Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,

train_df['Embarked1'] = train_df['Embarked'].map( {'C': 0, 'Q': 1, 'S':2} ).astype(int)
#train_df['Embarked2'] = train_df['Embarked'].map( {'C': 0, 'Q': 1, 'S':0} ).astype(int)



# TEST DATA
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.loc[(train_df.Embarked.isnull()), 'Embarked' ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df['Embarked1'] = test_df['Embarked'].map( {'C': 0, 'Q': 1, 'S':2} ).astype(int)
#test_df['Embarked2'] = test_df['Embarked'].map( {'C': 0, 'Q': 1, 'S':0} ).astype(int)





######### Age Missing Values

# Get title info
def name_info(row):
    if 'Miss.' in row['Name']:
        return 1
    elif 'Mrs.' in row['Name']:
        return 2
    elif 'Mr.' in row['Name']:
        return 3
    else:
        return 0    
         
train_df['title']=train_df.apply(name_info,axis=1)
test_df['title']=test_df.apply(name_info,axis=1)

all_df=pd.concat([train_df,test_df])

titles=all_df.title.unique()
title_age_mean=dict()
title_age_std=dict()
for t in titles:
    title_age_mean[t]=all_df[all_df['title']==t].Age.dropna().median()
    title_age_std[t]=all_df[all_df['title']==t].Age.dropna().std()

# Set age
def set_age(row):
    if pd.isnull(row['Age']):
        return np.random.normal(loc=title_age_mean[row['title']], scale=title_age_std[row['title']])   
    else:
        return row['Age']

train_df['Age']=train_df.apply(set_age,axis=1)
test_df['Age']=test_df.apply(set_age,axis=1)
  
    
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Embarked'], axis=1) 


# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Embarked'], axis=1) 


# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values



#****************
###Models
#0 Original Random Forests
from sklearn.ensemble import RandomForestClassifier
model0 = RandomForestClassifier(n_estimators=100)
model1 = RandomForestClassifier(n_estimators=200,min_samples_leaf=5,max_features=3)
models=[model0,model1]


#max_features
for i in range(7):
    modeli=RandomForestClassifier(n_estimators=200,min_samples_leaf=5+i*5,max_features=4)
    #models.append(modeli)
    
#1 Boosting
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier


for i in range(1):
    modeli=GradientBoostingClassifier(n_estimators=100, learning_rate=1, min_samples_leaf=5)
    models.append(modeli)


#models=[model0,model1,model2,model3,model4]

# Cross Validation
nFold=10
cv_index=np.random.random_integers(low=0, high=nFold-1, size=len(train_data))

X_all=train_data[0::,1::]
Y_all=train_data[0::,0]

modelN=len(models)

cv_mse=np.zeros(modelN)
cv_train_mse=np.zeros(modelN)

for i in range(nFold):
    X_train=X_all[cv_index!=i]
    y_train=Y_all[cv_index!=i]
    X_test=X_all[cv_index==i]
    y_test=Y_all[cv_index==i]
    
    for j in range(modelN):
        models[j].fit(X_train, y_train)
        y_pred=models[j].predict(X_test)
        cv_mse[j]+=np.sum(abs(y_test-y_pred))/len(y_test)/nFold    
        y_pred=models[j].predict(X_train)
        cv_train_mse[j]+=np.sum(abs(y_train-y_pred))/len(y_train)/nFold    
    
print cv_mse
print cv_train_mse


#****************
model_optimal=RandomForestClassifier(n_estimators=1000,min_samples_leaf=10,max_features=3)

print 'Training...'
model_optimal = model_optimal.fit( train_data[0::,1::], train_data[0::,0] )


print 'Predicting...'
output = model_optimal.predict(test_data).astype(int)


predictions_file = open("forest_adv_feature_adv_age.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
