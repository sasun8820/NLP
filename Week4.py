#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 17:56:34 2023

@author: jeonseo
"""

# Import function from Utils function
from Utils import *
import pandas as pd
import os 
import collections 

data_path = "/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/"

the_data = file_walker(data_path)

fishing = the_data[the_data.label == "fishing"]

str_cat = fishing.body.str.cat(sep=" ")  # sort of like a join function
        
wrd_freq = collections.Counter(str_cat.split())



test = file_reader(data_path + "fishing/0,4570,7-153-10364-34956--,00.html_121800529000.txt")
f = open(data_path + "Fishing-Reports_121810946000.txt", "r")

#readlines, readline, read
tmp = f.readlines()
tmp
f.close()

# read first 10 lines
tmp[7]

# read everything all at once in line
f = open(data_path + "Fishing-Reports_121810946000.txt", "r")

#readlines, readline, read
tmp = f.read()
tmp
f.close()


# clean text
def clean_txt(string):
    import re
    # clean up text, remove all special characters and lower case that text
    # 띄면 안됨!
    tmp_clean = re.sub(r"[^A-Za-z0-9]+|\[|\]" , " ", string).lower().strip()
    return tmp_clean 

print (clean_txt("cat# dog[, fun!"))



'''
build a function that takes in a full path to a file
and return a lower cased version with all special characters removed 
and only one space between tokens
'''


def file_reader(path_in):
    f = open(path_in, "r", encoding="UTF-8")
    tmp = f.read()
    tmp = clean_txt(tmp)
    f.close()
    return tmp 

path_in = data_path + "Fishing-Reports_121810946000.txt"
path_in
test = file_reader(path_in)
test



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
                txt_t = file_reader(path_t)   # path_t = txt file의 경로 
                if len(txt_t) > 0: # we will remove anything that has none in the file 
                    label_t = root.split('/')[-1:][0]
                    tmp_data = pd.DataFrame({'body': txt_t,
                                             'label': label_t}, index=[0])
                    t_data= pd.concat([t_data, tmp_data], ignore_index=True)
            except: 
                print(path_t)   # Let's see what file is having an issue 
                pass
    return t_data
    
the_data = file_walker(data_path)
                


'''
build a function called fun_wrd_cnt that returns a dictionary for EACH unique topic 
and whose value points to a dictionary whose keys are unique tokens 
and values are the frequency of the respective tokens 


topics = 4 topics in "data" folder
'''

def fun_wrd_cnt(pd_in):
    import collections
    fun_word = dict()
    for topic in pd_in.label.unique(): #fishing, hiking, machinelearning, mathematics..
        wrd_fun = pd_in[pd_in.label == topic]
        str_cat = wrd_fun.body.str.cat(sep=" ") # 위에 file_walker 가 DataFrame 리턴하는데
                                                # 그중 "Body" column 얘기하는거임
                                                # 일열로 하나로 세우는 것! 그래야 글자수 확인 가능.
        fun_word[topic] = collections.Counter(str_cat.split())
    return fun_word
        

wrd_dictionary = fun_wrd_cnt(the_data)
wrd_dictionary













