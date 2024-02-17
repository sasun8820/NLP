#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 22:03:26 2023

@author: jeonseo
"""

# Setting Working Directory 
import os
os.chdir('/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/')

# Importing functions we made in class 
from Utils import *


'''
#1) Please create a function called gen_senti that Tokenizes arbitrary text and compares each token with 
the positive and negative lexicons of each dictionary and outputs the sentiment score, S.  
Positive and negative words, pw and nw, count as a score of 1 and -1 respectively for each word matched.  
The total count for pw and nw are pc and nc, respectively.  Each message sentiment, S, is normalized between -1 and 1.  
Any text that does not any positive AND negative words would have to be ignored, and not scored. (60 points) 
where  For example: Let us say the following sentence was an input into the function “The darkest hour is among us 
in this time of gloom, however, we will prevail!”.  Let’s pretend the negative words were darkest and gloom and 
positive words were prevail

S = (-1 + -1 + 1) / 3 = -1/3 = -0.3333
'''

# Reading the text files 
pos = 'positive-words.txt' 
neg = 'negative-words.txt'
 
with open(pos, 'r') as pw:
    pw = pw.read().split()
    print(pw)
    
    
    
with open(neg, 'r', encoding='latin-1') as nw:
    nw = nw.read().split()
    print(nw)
    


def clean_txt(str_in):
    import re
    tmp_clean_t = re.sub("[^A-Za-z'-]+", " ", str_in
                         ).lower().strip()
    return tmp_clean_t


def gen_senti(text):
    words = clean_txt(text)
    words = words.split()

    pc = 0
    nc = 0
    score = 0 # sentiment score
    
    for word in words:
        if word in pw:
        # if word in pw and pw.count(word) == 1:
            pc += 1
            score += 1

        elif word in nw:
            nc += 1
            score -= 1 
        
        else:
            score += 0
      
    if pc > 0 or nc > 0:
        S = score/ (pc+nc)
        return S 
    
    return 0


gen_senti("time-consuming") # -1 
gen_senti("successfully") # +1
gen_senti("this task was time-consuming but has been done successfully.") # (-1+1) / 2 = 0
gen_senti("fish never fails") # (0+0-1) / 1 = -1
gen_senti("We went to fishing, but we failed") # (0+0...-1) / 1 = -1



# text = "get your deep sea fishing trip tickets daveys locker home whale watching cruises whale watching prices times whale watching field trips whale dolphin sightings whale watching gallery reserve your trip deep sea fishing trips deep sea fishing prices times overnight deep sea fishing fishing boat rentals dl fishing gallery whale watching cruise tickets deep sea fishing trip tickets deep sea fishing long beach deep sea fishing los angeles deep sea fishing newport beach deep sea fishing orange county deep sea fishing catalina island whale watching dana point visitors whale watching long beach visitors whale watching los angeles visitors whale watching laguna beach visitors whale watching san diego visitors whale watching orange county burial at sea christmas boat parade fish count reserve a rental fishing boat rentals six pack boat rental skiff rental electric boat rental yacht cruise ship rental company fishing trip contact deep sea fishing trip tickets enter pre purchased tickets coupons vouchers in voucher code box last page of checkout enter pre purchased tickets coupons vouchers in voucher code box last page of checkout trip time sold out please call for reservation options search davey s locker search davey s locker search for join the davey s locker email club join the davey s locker email club davey's locker"
    
# okay = clean_txt(text).split()

# pos_score = 0
# for o in okay:
#     if o in pw:
#         pos_score += 1
        
# print(pos_score)

# pos_count = 0
# for o in okay:
#     if o in pw:
#         pos_count += 1
        
# print(pos_count)

# neg_score = 0
# for o in okay:
#     if o in nw:
#         neg_score -= 1
        
# print(neg_score)

# neg_count = 0
# for o in okay:
#     if o in nw:
#         neg_count += 1
        
# print(neg_count)

# print((pos_score + neg_score) / (pos_count + neg_count))

        


'''
#2) 
Using the dataframe, body column, from lecture, the_data, apply this function to each corpus and 
add a column called “simple_senti” (15 points)
'''

data_path = "/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/data/"

def file_walker(data_path):
    import os
    import pandas as pd 
    t_data = pd.DataFrame()
    # root: direcotry
    # dirs: folder
    # files: files
    
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            path_t = root + "/" + name 
            try:
                txt_t = file_reader(path_t)  
                if len(txt_t) > 
                    label_t = root.split('/')[-1:][0]
                    tmp_data = pd.DataFrame({'body': txt_t,
                                             'label': label_t}, index=[0])
                    t_data= pd.concat([t_data, tmp_data], ignore_index=True)
            except: 
                print(path_t)  
                pass
    return t_data

the_data = file_walker(data_path)

the_data['simple_senti'] = the_data.body.apply(gen_senti)



'''
#3) 
Using vaderSentiment, apply the “compound” value of sentiment for each corpus of body column, 
on a new column of the_data called “vader” (15 points)
'''

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def compound_score(text):
    
    analyzer = SentimentIntensityAnalyzer()
    
    text = clean_txt(text) # Spliting would cause inaccuracy with VADER
    
    sentiment = analyzer.polarity_scores(text)
    
    return sentiment['compound']

the_data['vader'] = the_data.body.apply(compound_score)

    

'''
#4)
Compute the mean, median and standard_deviations of both sentiment measures, 
“simple_senti” and “vader”  (10 points)
'''

def compute(column1, column2):
    import numpy as np
    import pandas as pd

    mean1 = np.mean(column1)
    median1 = np.median(column1)
    std_dev1 = np.std(column1)

    mean2 = np.mean(column2)
    median2 = np.median(column2)
    std_dev2 = np.std(column2)

    compute_df = pd.DataFrame({
        'Statistic': ['Mean', 'Median', 'Standard Deviation'],
        'simple_senti': [mean1, median1, std_dev1],
        'vader': [mean2, median2, std_dev2]
    })

    return compute_df

values = compute(the_data['simple_senti'], the_data['vader'])
values
    
    

    













