#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 00:47:18 2023

@author: jeonseo
"""

''' 
<Homework 3>

Build a function, called word_prob, that outputs probabilities, for every possible topic 
[all, fishing, hiking, machinelearning, mathematics] that a token or  sequential token combination 
shows up in an arbitrary textual based column  (body, body_sw, body_sw_stem), dictated by the user, from the dataframe, the_data, 
we have been using  in class.  The output dictionary of the function needs to have the following keys:


all: <probability the sequential input token(s) shows up in ALL the corpuses

fishing: <probability the sequential input token(s) shows up in the fishing corpuses

hiking: <probability the sequential input token(s) shows up in the hiking corpuses

machinelearning: <probability the sequential input token(s) shows up in the machinelearning corpuses

mathematics: <probability the sequential input token(s) shows up in the mathematics corpuses

 

The 'value' field of a dictionary is to have a value of None if the token(s) do not show up

NOTE: If there are a total of 100 tokens, and the count of a specific token is 12, the probability of that token showing up is 12/100=.12
'''

import os
os.chdir('/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/')

from Utils import *

data_path = "/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/data/"

the_data = file_walker(data_path)

the_data['body_sw'] = the_data.body.apply(rem_sw)

the_data['body_cnt'] = the_data.body.apply(token_counts)
the_data['body_sw_cnt'] = the_data.body_sw.apply(token_counts)



the_data['body_cnt_u'] = the_data.body.apply(token_counts_unique)
the_data['body_sw_cnt_u'] = the_data.body_sw.apply(token_counts_unique)


the_data["body_sw_stem"] = the_data.body_sw.apply(stem_fun)
the_data["body_sw_stem_cnt"] = the_data.body_sw_stem.apply(token_counts)
the_data["body_sw_stem_cnt_u"] = the_data.body_sw_stem.apply(token_counts_unique)



def word_prob(dataframe, tokens, column_names):
    
    probabilities = {
        "all": None,
        "fishing": None,
        "hiking": None,
        "machinelearning": None,
        "mathematics": None
    }

    def clean_txt(str_in):
        import re
        tmp_clean_t = re.sub("[^A-Za-z'-]+", " ", str_in).lower().strip()
        return tmp_clean_t

    # Tokenize the input tokens into individual words
    tokens = clean_txt(tokens).split()

    token_counts_all = {label: 0 for label in probabilities.keys()}
    
    # Total number of tokens in a selected column
    total_tokens = dataframe[column_names].apply(lambda x: ' '.join(x)).str.split().apply(len).sum()

 
    for corpus in probabilities.keys():  # all, fishing, hiking, machinelearning, mathematics
        if corpus != "all":
            label_data = dataframe[dataframe["label"] == corpus]    # Sort by "Label"
            label_text = ' '.join(label_data[column_names])         
            
            # Split them into individual words
            label_words = label_text.split()
            
            token_counts = 0
            
            # Loop including the multi-words 
            for i in range(len(label_words) - len(tokens) + 1): # Until the last block of our input tokens!
                if label_words[i: i+len(tokens)] == tokens:
                    token_counts += 1

            if token_counts > 0:
                probabilities[corpus] = token_counts / total_tokens
    
    # Codes for "All" corpus         
    labels_with_token = [corpus for corpus in probabilities.keys() if corpus != "all" and probabilities[corpus] is not None]

    if len(labels_with_token) == 4:  # When a word is in all four labels 
        tokens_in_all_labels = sum(probabilities[label] for label in labels_with_token)
        probabilities["all"] = tokens_in_all_labels / total_tokens

    return probabilities


print(word_prob(the_data, "hiking", "body"))
print(word_prob(the_data, "fishing kdwpt kdwpt", "body"))








