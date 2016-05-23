# This script only focuses on feature engineering
# This script generates and saves a feature matrix, feature name, etc, in pickle
# The output will be picked up by different models to fit and predict

#V6: Spell Patch

import __future__
import time
start_time = time.time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as pickle
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn import pipeline, grid_search
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from nltk.stem.porter import *
import re
stemmer = PorterStemmer()

#Typo Correction
import os
Dir_lib='/Users/zhenyuz/Documents/Projects/Kaggle/HomeDepot'
os.chdir(Dir_lib)
import kungfu_model_patch_spell_correct as lib_sc

############### Function Part #################
#String stemmer
def str_stem(s):
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        #s = re.sub(r'( [a-z]+)([A-Z][a-z])', r'\1 \2', s)
        s = s.lower()
        s = s.replace("  "," ")
        s = re.sub(r"([0-9]),([0-9])", r"\1\2", s)
        s = s.replace(","," ")
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        #s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])

        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air condition")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"


#A Feature Generating Function
#Input: (1)Feature Name, (2)Field in Attribute for database, (3) Fields in Attribute for matching (4)Value remove List,
#Output: (1) Data Frame column: Feature: match 1, non match -1, na 0; (2) Data Frame column: Fields in Attribute for matching
### Feature Engine V2
def attr_base_match(query, attr_base):
    if sum(int(word in attr_base) for word in query.split())>0:
        return 1
    else:
        return 0


### Generate 4 Cols: query - base match 0/1 whole_word, query - attr match common_word, attr, len of attr
def attr_feature_engine(df_all,df_attr,feature_name,attr_col_base,attr_col_match,remove_values=[],term_base=False):
    ###Step 1:  Create word data base for identifying whether feature appears in search term
    #Get the attribute for word data base
    df_attr_base=df_attr[df_attr['name']==attr_col_base].reset_index(True)
    #Create color word base
    color_base1=df_attr_base.value.unique()
    color_base2=color_base1
    if not term_base:
        color_base3=[]
        for i in range(len(color_base2)):
            color_base3+=color_base2[i].split()
        color_base3=list(set(color_base3))
    else:
        color_base3=color_base2
    #Remove undesired values
    for i in range(len(remove_values)):
        if remove_values[i] in color_base3:
            color_base3.remove(remove_values[i])
    ###Step 2: Set up matching column in df_all
    df_attr_match=None
    for i in range(len(attr_col_match)):
        df_attr_matchi = df_attr[df_attr.name == attr_col_match[i]][["product_uid", "value"]].rename(columns={"value": attr_col_match[i]})
        if df_attr_match is None:
            df_attr_match=df_attr_matchi
        else:
            df_attr_match=pd.merge(df_attr_match, df_attr_matchi, how='outer', on='product_uid')

    df_attr_match=df_attr_match.drop_duplicates(subset=['product_uid']).reset_index(drop=True)
    #Create a combined column for matching
    df_attr_match=df_attr_match.fillna('')
    df_attr_match[feature_name+'_attr']=df_attr_match[attr_col_match[0]]
    i=1
    while i<len(attr_col_match):
        df_attr_match[feature_name+'_attr']=df_attr_match[feature_name+'_attr']+' '+df_attr_match[attr_col_match[i]]
        i+=1
    #df_attr_match[feature_name+'_attr']=df_attr_match[feature_name+'_attr'].map(lambda x:x.decode('utf-8'))
    #df_attr_match[feature_name+'_attr']=df_attr_match[feature_name+'_attr'].map(lambda x:str_stemmer2(x))
    #delete duplicates
    df_attr_match[feature_name+'_attr']=[' '.join(set(s.split(' '))) for s in df_attr_match[feature_name+'_attr']]
    #Join into df_all
    if feature_name+'_attr' in df_all.columns:
        df_all=df_all.drop(feature_name+'_attr',1)
    df_all=pd.merge(df_all,df_attr_match[['product_uid',feature_name+'_attr']], how='left',on='product_uid')
    df_all[feature_name+'_attr']=df_all[feature_name+'_attr'].fillna('')
    ###Step 3: Create Feature Column in df_all
    df_all['query_feature_match'] = df_all['search_term']+"\t"+df_all[feature_name+'_attr']
    #query - base
    df_all[feature_name+'_show']=df_all['query_feature_match'].map(lambda x:attr_base_match(x.split('\t')[0],color_base3))
    #query - attr
    df_all[feature_name+'_match']=df_all['query_feature_match'].map(lambda x: str_common_word(x.split('\t')[0],x.split('\t')[1]))
    #len of feature attr
    df_all['len_of_'+feature_name+'_attr']=df_all[feature_name+'_attr'].map(lambda x:len(x.split())).astype(np.int64)
    df_all=df_all.drop('query_feature_match',1)
    return df_all



### Word matching
#Number of whole words matches
def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

#Number of word matches
def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())

#Number of Shringles matches
def str_common_block(str1, str2):
    lstr2=str2.split(' ')
    return sum(lstr2.count(word) for word in str1.split())

# Creating Shringles
def mblocks(str1,window):

    str1=str1.split(' ')
    res=[]

    for i in range(len(str1)):
        if len(str1[i])<=window:
            res.append(str1[i])
        else:
            res1=[]
            j=window
            while j<=len(str1[i]):
                res1.append(str1[i][(j-window):j])
                j+=1
            res+=res1
    return " ".join(res)


# Dealing with Unicode
def unicode_fix(X_t):
    if isinstance(X_t['search_term'].values[0],unicode):
        X_t['search_term']=X_t['search_term'].map(lambda x: normalize('NFKD', x).encode('ASCII', 'ignore'))
    if isinstance(X_t['product_title'].values[0],unicode):
        X_t['product_title']=X_t['product_title'].map(lambda x: normalize('NFKD', x).encode('ASCII', 'ignore'))
    if isinstance(X_t['product_description'].values[0],unicode):
        X_t['product_description']=X_t['product_description'].map(lambda x: normalize('NFKD', x).encode('ASCII', 'ignore'))
    if isinstance(X_t['product_info'].values[0],unicode):
        X_t['product_info']=X_t['product_info'].map(lambda x: normalize('NFKD', x).encode('ASCII', 'ignore'))

    fe_attr=['brand','fe_brand_attr','fe_color_attr','fe_material_attr','query_brand','attr_combo','product_attr_blocks','query_attr_blocks']
    for attr_i in range(len(fe_attr)):
        if isinstance(X_t[fe_attr[attr_i]].values[0],unicode):
            brand_str=[]
            for i in X_t[fe_attr[attr_i]].values:
                if isinstance(i,unicode):
                    brand_str.append(normalize('NFKD', i).encode('ASCII', 'ignore'))
                else:
                    brand_str.append(i)
            X_t[fe_attr[attr_i]]=brand_str
    return X_t






#Parameters
stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':0}


########################### Read Data
# Load the data into DataFrames
Dir = '/Users/zhenyuz/Documents/Projects/Kaggle/HomeDepot/data'

stemmer = SnowballStemmer('english')

df_train = pd.read_csv('%s/train.csv'%(Dir), encoding="ISO-8859-1")
df_test = pd.read_csv('%s/test.csv'%(Dir), encoding="ISO-8859-1")
df_attr = pd.read_csv('%s/attributes.csv'%(Dir))
df_pro_desc = pd.read_csv('%s/product_descriptions.csv'%(Dir))

num_train = df_train.shape[0]

### Standard Word Matching
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

### Typo Correction for Search Terms
searches=df_all['search_term'].values
searches_adjust=[]
search_typo=[]
for search in searches:
    if search in lib_sc.spell_check_dict:
        searches_adjust.append(lib_sc.spell_check_dict[search])
        search_typo.append(1)
    else:
        searches_adjust.append(search)
        search_typo.append(0)
df_all.loc[:,'search_term']=searches_adjust
df_all.loc[:,'search_typo']=search_typo


#Unit transform
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
#Stemmer
#df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
#df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
#df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
#Len of query/title/description
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_title_ratio'] = df_all['query_in_title']/df_all['len_of_title']
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_title_ratio'] = df_all['word_in_title']/df_all['len_of_title']

df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
df_all['query_in_description_ratio'] = df_all['query_in_description']/df_all['len_of_description']
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['word_in_description_ratio'] = df_all['word_in_description']/df_all['len_of_description']



### SAVE Processed Data
pickle_file = '%s/df_v6A_1.pickle'%(Dir)
try:
  f = open(pickle_file, 'wb')
  save = {
    'df_all': df_all,
    }
  #pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(save, f)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
### SAVED
### Load
pickle_file = '%s/df_v6A_1.pickle'%(Dir)
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  df_all = save['df_all']
  del save  # hint to help gc free up memory
  print('df_all', df_all.shape)
##########################


### Attr
df_attr['value']=df_attr['value'].map(lambda x:str_stem(x))

df_attr_gb=df_attr.groupby(['product_uid']).agg({'value': lambda x: ' '.join(x)})
df_attr_gb=df_attr_gb.reset_index()
df_attr_gb.rename(columns={'value':'attr_combo'}, inplace=True)
# Merge into all
df_all = pd.merge(df_all, df_attr_gb, how='left', on='product_uid')
df_all['attr_combo']=df_all['attr_combo'].fillna('')
#Standard Word Matching
df_all['len_of_attr'] = df_all['attr_combo'].map(lambda x:len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']+"\t"+df_all['attr_combo']
df_all['query_in_attr'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[3],0))
df_all['word_in_attr'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[3]))
df_all['word_in_attr_ratio'] = df_all['word_in_attr']/(df_all['len_of_query']+df_all['len_of_attr'])
df_all['query_in_attr_ratio'] = df_all['query_in_attr']/(df_all['len_of_query']+df_all['len_of_attr'])

## Add len char feature
df_all['len_of_query_char']=[len(x) for x in df_all['search_term'].values]
df_all['len_of_title_char']=[len(x) for x in df_all['product_title'].values]
df_all['len_of_description_char']=[len(x) for x in df_all['product_description'].values]
df_all['len_of_attr_char']=[len(x) for x in df_all['attr_combo'].values]




### Shingling and Jaccard Similarity
## Input df,Field1, Field2, Shingle size
## Output a List of Jaccard Similarity Only
def jaccard_sim(dfx,col1,col2,sgsize):
    df1=dfx[[col1,col2]]
    #Shingling query
    df1.loc[:,col1+'blocks'] = df1[col1].map(lambda x: mblocks(x,sgsize))
    df1.loc[:,'len_of_'+col1+'_blocks'] = df1[col1+'blocks'].map(lambda x:len(x.split())).astype(np.int64)
    #Shingling title
    df1.loc[:,col2+'blocks'] = df1[col2].map(lambda x: mblocks(x,sgsize))
    df1.loc[:,'len_of_'+col2+'_blocks'] = df1[col2+'blocks'].map(lambda x:len(x.split())).astype(np.int64)
    #Join query and title
    df1.loc[:,'join_blocks'] = df1[col1+'blocks']+"\t"+df1[col2+'blocks']
    df1.loc[:,'len_of_common_blocks'] = df1['join_blocks'].map(lambda x:str_common_block(x.split('\t')[0],x.split('\t')[1]))
    #Jaccard similarity
    df1.loc[:,col1+'_'+col2+'_jaccard%s'%(sgsize)]=df1['len_of_common_blocks']/(df1['len_of_'+col1+'_blocks']+df1['len_of_'+col2+'_blocks'])
    return df1[col1+'_'+col2+'_jaccard%s'%(sgsize)].values


sgsizes=[2,3,4,5,6,7,8,9,10]
col1s=['search_term']
col2s=['product_title','product_description','attr_combo']
for col1 in col1s:
    for col2 in col2s:
        for sgsize in sgsizes:
            df_all.loc[:,col1+'_'+col2+'_jaccard%s'%(sgsize)]=jaccard_sim(dfx=df_all,col1=col1,col2=col2,sgsize=sgsize)

### SAVE Processed Data
pickle_file = '%s/df_v6A_2.pickle'%(Dir)
try:
  f = open(pickle_file, 'wb')
  save = {
    'df_all': df_all,
    }
  #pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(save, f)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
### SAVED
### Load
pickle_file = '%s/df_v6A_2.pickle'%(Dir)
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  df_all = save['df_all']
  del save  # hint to help gc free up memory
  print('df_all', df_all.shape)
##########################


### Attributes Features

# Color
df_all=attr_feature_engine(df_all,df_attr,feature_name='fe_color',attr_col_base='Color Family',attr_col_match=['Color Family','Color/Finish'],remove_values=['unfinish','custom','medium','.combin','multi'],term_base=False)

# Material
df_all=attr_feature_engine(df_all,df_attr,feature_name='fe_material',attr_col_base='Material',attr_col_match=['Material'],term_base=False)

# Brand Feature
df_all=attr_feature_engine(df_all,df_attr,feature_name='fe_brand',attr_col_base="MFG Brand Name",attr_col_match=["MFG Brand Name"],remove_values=[],term_base=True)


### SAVE Processed Data
pickle_file = '%s/df_v6A_3.pickle'%(Dir)
try:
  f = open(pickle_file, 'wb')
  save = {
    'df_all': df_all,
    }
  #pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(save, f)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
### SAVED
### Load
pickle_file = '%s/df_v6A_3.pickle'%(Dir)
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  df_all = save['df_all']
  del save  # hint to help gc free up memory
  print('df_all', df_all.shape)
##########################






### Tfidf Features
#Input Text Feature, SVD_Cols
#Output SVD Cols for this feature
def tfidf_svd(colx,svdn):
    tvec=TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_w)
    tvec_mx=tvec.fit_transform(df_all[colx])
    tsvd = TruncatedSVD(n_components=svdn, random_state = 2009)
    tsvd_output=tsvd.fit_transform(tvec_mx)
    return tsvd_output

text_features=['search_term','product_title','product_description','attr_combo']
svdns=[30,50,50,10]
for i in range(len(text_features)):
    tsvd_mx=tfidf_svd(text_features[i],svdns[i])
    tfidf_names=[text_features[i]+'_tsvd_'+str(j) for j in range(svdns[i])]
    for k in range(len(tfidf_names)):
        df_all.loc[:,tfidf_names[k]]=[x[k] for x in tsvd_mx]


### SAVE Processed Data
pickle_file = '%s/df_v6A_4.pickle'%(Dir)
try:
  f = open(pickle_file, 'wb')
  save = {
    'df_all': df_all,
    }
  #pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(save, f)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
### SAVED
### Load
pickle_file = '%s/df_v6A_4.pickle'%(Dir)
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  df_all = save['df_all']
  del save  # hint to help gc free up memory
  print('df_all', df_all.shape)
##########################



####### N-gram
import ngram
#ngram.NGram.compare(df_all['search_term'][0],df_all['product_title'][0],N=3)
sgsizes=[1,2,3,4,5,6,7,8,9,10]
col1s=['search_term']
col2s=['product_title','product_description','attr_combo']
for col1i in range(len(col1s)):
    for col2i in range(len(col2s)):
        for sgsize in sgsizes:
            df_all.loc[:,col1s[col1i]+'_'+col2s[col2i]+'_ngram%s'%(sgsize)]=df_all['product_info'].map(lambda x:ngram.NGram.compare(x.split('\t')[0],x.split('\t')[col2i+1],N=sgsize))



### SAVE Processed Data
pickle_file = '%s/df_v6A_5.pickle'%(Dir)
try:
  f = open(pickle_file, 'wb')
  save = {
    'df_all': df_all,
    }
  #pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(save, f)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
### SAVED
### Load
pickle_file = '%s/df_v6A_5.pickle'%(Dir)
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  df_all = save['df_all']
  del save  # hint to help gc free up memory
  print('df_all', df_all.shape)
##########################





###Clean Up

### Drop Text Columns, and Store Feature Names
[print(i, df_all.columns[i], df_all[df_all.columns[i]][0]) for i in range(len(df_all.columns))]

char_column_names=[]
for i in range(len(df_all.columns)):
    if isinstance(df_all[df_all.columns[i]][0], str):
        print(i, df_all.columns[i], df_all[df_all.columns[i]][0])
        char_column_names.append(df_all.columns[i])

cols_names_drop=['product_title', 'search_term', 'product_description', 'product_info','attr_combo', 'fe_color_attr', 'fe_material_attr', 'fe_brand_attr']

df_all=df_all.drop(cols_names_drop,1)

### SAVE Processed Data
pickle_file = '%s/df_v6A_6.pickle'%(Dir)
try:
  f = open(pickle_file, 'wb')
  save = {
    'df_all': df_all,
    }
  #pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(save, f)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
### SAVED

### Load
pickle_file = '%s/df_v6A_6.pickle'%(Dir)
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  df_all = save['df_all']
  del save  # hint to help gc free up memory
  print('df_all', df_all.shape)
##########################















### Plot the Histogram of product uid
pids=df_all['product_uid'].values

fig = plt.figure(figsize=(16.0, 10.0))

fig.add_subplot(111)

plt.hist(pids,bins=20,color='grey')

### Plot relevance vs. product uid
df_pids_score=df_train.groupby(['product_uid']).relevance.mean().reset_index(True)

df_pids_score['pid_short']=[int(str(x)[:-1]) for x in df_pids_score['product_uid'].values]
df_pids_score['pid_short']

pids=df_pids_score['product_uid'].values
relevance=df_pids_score['relevance'].values

#ax.plot(pids,relevance,'ro')


from scipy.interpolate import interp1d
f2 = interp1d(pids[:10000], relevance[:10000], kind='cubic')

fig = plt.figure(figsize=(16.0, 10.0))

ax=fig.add_subplot(111)

xnew = np.linspace(min(pids[:10000]), max(pids[:10000]), num=100, endpoint=True)
ax.plot( xnew, f2(xnew), '--')
ax.set_ylim([0.,4.])












#Word Count
from collections import Counter
r1=df_all['search_term'].values
word_count=Counter(" ".join(r1).split(" ")).items()

top_word=[]
top_count=[]
for key, value in word_count:
    if value>1000:
        top_word.append(key)
        top_count.append(value)

sort_index=sorted(range(len(top_count)), key=lambda k: top_count[k])

sorted_list=[(top_word[i],top_count[i]) for i in sort_index]

sorted_list=sorted_list[::-1]











