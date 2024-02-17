#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 19:02:27 2023

@author: jeonseo
"""

"""
- py stands for the Python File

- To run all the codes, hit the "PLAY" button on top

- fn + 9: Run the selected code

- fn + 5: Run the entire code

- cmd + 1: # 

"""

from Utils import *

# Stopwords: a, the, an, etc 

import nltk
sw = nltk.corpus.stopwords.words("english")

# Stopwords could be: 've, up, for, so... etc 
sent = "i have been standing up for so long that I'm tired"

# 1. Tokenize 
new_ar = []
for word in sent.lower().split():
    # Filtering out if the word is not a stopword 
    if word not in sw:
        new_ar.append(word)


# Join Array of Tokens (한줄로 만들기!)
sent = ' '.join(new_ar)
sent


# List of words that are not stopwords 
sent = [word for word in sent.lower().split() if word not in sw]
sent = ' '.join(new_ar)
sent


'''
Create a function called rem_sw that takes a string input 
and output a string any tokens that show up in SW removed
'''

def rem_sw(str_in): # word 
    import nltk
    sw = nltk.corpus.stopwords.words("english")
    sent = [word for word in str_in.lower().split() if word not in sw]
    sent = ' '.join(sent)
    return sent 
            
            
    
'''
Create a new column to the_data called body_sw
where each corpus along column body, on each row, has stopwords removed
'''    
data_path = "/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/data/"

the_data = file_walker(data_path)

# apply(function) to certain column
the_data['body_sw'] = the_data.body.apply(rem_sw)




 
''' 
Create a function that counts the total number of tokens 
and outputs the total number of tokens 
apply this function to both body and body_sw, call the new columns 
columns body_cnt and body_sw_cnt, respectively
'''

def token_counts(str_in): 
    tmp = str_in.lower().split()
    return len(tmp)
    

the_data['body_cnt'] = the_data.body.apply(token_counts)
the_data['body_sw_cnt'] = the_data.body_sw.apply(token_counts)



the_data['body_cnt_u'] = the_data.body.apply(token_counts_unique)
the_data['body_sw_cnt_u'] = the_data.body_sw.apply(token_counts_unique)




































    
    
    
