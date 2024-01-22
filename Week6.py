==#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:59:08 2023

@author: jeonseo
"""

"""
- py stands for the Python File

- To run all the codes, hit the "PLAY" button on top

- fn + 9: Run the selected code

- fn + 5: Run the entire code

- cmd + 1: # 


- # Get the current working directory 
import os
print(os.getcwd())

- # Change the working directory 
import os
os.chdir('/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/')

"""

 

'''
 tmp_pd = tmp_pd.append({'body': tmp_txt,
                         'label': re.sub(' ', '_', q_blah)
                         }, ignore_index=True)


1. Fix the above code  in crawler.py to perform a proper concatenation and also 
implement the clean_txt function before populating the tmp_pd (Data frame)
which is the output of write_crawl_results 


2. Replace file_walker with the output of write_crawl_results

'''

from Utils import *

# 1)

the_data = write_crawl_results(["taylor swift", "pink flyod"], 20)
the_data

the_data['body_sw'] = the_data.body.apply(rem_sw)


the_data['body_cnt'] = the_data.body.apply(token_counts)
the_data['body_sw_cnt'] = the_data.body_sw.apply(token_counts)



the_data['body_cnt_u'] = the_data.body.apply(token_counts_unique)
the_data['body_sw_cnt_u'] = the_data.body_sw.apply(token_counts_unique)


wrd_dictionary = fun_wrd_cnt(the_data)



import pickle


# Dump object(Data) to that file 
pickle.dump(the_data, open(out_path + "the_data.pk", "wb"))  # wb: write mode
 

the_data = pickle.load(open(out_path + 'the_data.pk', 'rb')) # rb: read mode

write_pickle(the_data, out_path, "the_data")

'''
1. Create a function called read_pickle that reads 
in the file in some location, aribitrary path and file_name


2. Create a function called write_pickle that writes out 
any python object to some location, arbitrary path, and file_name
'''

def write_pickle(obj_in, path_in, file_name):
    import pickle
    pickle.dump(obj_in, open(path_in + file_name + '.pk', 'wb'))
    
write_pickle(the_data, out_path, 'the_data_swift')    


def read_pickle(path_in, file_name):
    import pickle
    tmp_o = pickle.load(
        open(path_in + file_name +'.pk', 'rb'))
    return tmp_o

the_data = read_pickle(out_path, "the_data")
    
    
    
    
'''
Stemming algoritm: Reducing words to their roots 
'''

import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer()
ex = "after I went on a hike, I went fishing but caught nothing"

ps.stem("fishing")
ps.stem("weekly")





''' 
Create a function called stem_fun that takes in an arbitrary corpus
and outputs a stemmed version of that corpus 
'''
    
def stem_fun(str_in):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    sent = [ps.stem(word) for word in str_in.lower().split()]
    sent = ' '.join(sent)
    return sent
    

the_data["body_sw_stem"] = the_data.body_sw.apply(stem_fun)
the_data["body_sw_stem_cnt"] = the_data.body_sw_stem.apply(token_counts)
the_data["body_sw_stem_cnt_u"] = the_data.body_sw_stem.apply(token_counts_unique)



the_data = read_pickle(out_path, "the_data")



# Run vadersentiment through each columns 




