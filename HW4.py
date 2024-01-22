#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:23:31 2023

@author: jeonseo
"""

## Set Working Directory & Bring Util Functions ##
     
import os
os.chdir('/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/')
from Utils import *
path = "/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/"

## Scripts from my_reddit ## 

import praw
import pandas as pd
import pytz
from datetime import datetime

subreddit_channel = 'politics'

reddit = praw.Reddit(
     client_id="qAVCialmVTLXjAY0_YDttA",
     client_secret="qLxKfi7iQU1CtIoBfngh5uMvZyammg",
     user_agent="testscript by u/fakebot3",
     username="sasun8820",
     password="Wjstj2327!",
     check_for_async=False
 )

print(reddit.read_only)

def conv_time(var):
    tmp_df = pd.DataFrame()
    tmp_df = tmp_df.append(
        {'created_at': var}, ignore_index=True)
    tmp_df.created_at = pd.to_datetime(
        tmp_df.created_at, unit='s').dt.tz_localize(
            'utc').dt.tz_convert('US/Eastern') 
    return datetime.fromtimestamp(var).astimezone(pytz.utc)

def get_reddit_data(var_in):
    tmp_dict = pd.DataFrame()
    tmp_time = None
    try:
        tmp_dict = tmp_dict.append({"created_at": conv_time(
            var_in.created_utc)}, ignore_index=True)
        tmp_time = tmp_dict.created_at[0]
    except:
        print("ERROR")
        pass
    tmp_dict = {'msg_id': str(var_in.id),
                'author': str(var_in.author),
                'body': var_in.body, 'datetime': tmp_time}
    return tmp_dict


## Loading Pickle Files ##
tmp_vec = read_pickle(path, "vectorizer")
tmp_pca = read_pickle(path, "pca")
tmp_model = read_pickle(path, "my_model")          


## Modifying my_reddit Function Including Loaded Pickle Files ## 
for comment in reddit.subreddit(subreddit_channel).stream.comments():
    tmp_df = get_reddit_data(comment)
    cleaned_text = clean_txt(tmp_df["body"])
    without_sw = rem_sw(cleaned_text)
    stemmed_text = stem_fun(without_sw)
    vec_txt = tmp_vec.transform([stemmed_text]).toarray()
    pca_txt = tmp_pca.transform(vec_txt)

    y_pred = tmp_model.predict(pca_txt)
    predict_prob = tmp_model.predict_proba(pca_txt)[0]
    result_df = pd.DataFrame(predict_prob, index=tmp_model.classes_, columns=["probability"])
    
    print(f"Class Label Prediction is {y_pred}\n Likelihood Score is ")
    print(result_df)