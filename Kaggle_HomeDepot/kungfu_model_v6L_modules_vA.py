#Contains modules

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


