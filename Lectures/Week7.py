#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 18:03:27 2023

@author: jeonseo

- py stands for the Python File

- To run all the codes, hit the "PLAY" button on top

- fn + 9: Run the selected code

- fn + 5: Run the entire code

- cmd + 1: # 

"""
import os
os.chdir('/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/')

from Utils import *

data_path = "/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/data/"

the_data = file_walker(data_path)

#  Vectorization: Transforming text-based data into numeric values 

'''
data_path = "/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/data/"

the_data = file_walker(data_path)

# apply(function) to certain column
the_data['body_sw'] = the_data.body.apply(rem_sw)

the_data['body_cnt'] = the_data.body.apply(token_counts)
the_data['body_sw_cnt'] = the_data.body_sw.apply(token_counts)



the_data['body_cnt_u'] = the_data.body.apply(token_counts_unique)
the_data['body_sw_cnt_u'] = the_data.body_sw.apply(token_counts_unique)


the_data["body_sw_stem"] = the_data.body_sw.apply(stem_fun)
the_data["body_sw_stem_cnt"] = the_data.body_sw_stem.apply(token_counts)
the_data["body_sw_stem_cnt_u"] = the_data.body_sw_stem.apply(token_counts_unique)

'''



'''
Term-frequency 


CountVectorizer 

'''


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 

cv = CountVectorizer(ngram_range=(1,1))# give me all unigram 

#xform_data = cv.fit_transform(the_data.body).toarray()
xform_data = pd.DataFrame(cv.fit_transform(the_data.body).toarray())

# Apply column names on the above dataframe 
xform_data.columns = cv.get_feature_names_out()
# Apply labels 
xform_data.index = the_data.label

# 16718 unique tokens = 16718 unigrams 


# See if such sequential words (mathematics department) exist and persist 
#test = xform_data["mathematics department"]




''' 
Build a function called vec_fun, that takes a column of pandas df 
and vectorizes the data, the function needs to return the vectorized dataframe
the function needs to save off the cv object and you need to allow the user
to input custom ngram_range (m,n)
'''

def vec_fun(df_in, path_in, name_in, m_in, n_in, label_in):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd 
    
    cv = CountVectorizer(ngram_range=(m_in,n_in))
    xform_data_t = pd.DataFrame(cv.fit_transform(df_in).toarray())
    xform_data_t.columns = cv.get_feature_names_out()
    xform_data_t.index = label_in
    
    # Save it in my out_path! 
    write_pickle(cv, path_in, name_in)
    return xform_data_t



vec_data = vec_fun(the_data.body, out_path, "vec", 1, 1, the_data.label)

#vec_data = vec_fun(the_data.body_sw_stem, out_path, "vec", 1, 1, the_data.label)



'''
Term-Frequency Inverse Document Frequency 

Weight = frequency * log(N/df)

-	N: Total documents
-	df: in how many documents the certain letter appears 
-	Letters like “the” which appears in every document will return 0. 
    (“Let’s not put much weight on this word!”
'''

def vec_fun(df_in, path_in, name_in, m_in, n_in, label_in):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd 
    
    cv = TfidfVectorizer(ngram_range=(m_in,n_in))
    xform_data_t = pd.DataFrame(cv.fit_transform(df_in).toarray())
    xform_data_t.columns = cv.get_feature_names_out()
    xform_data_t.index = label_in
    
    # Save it in my out_path! 
    write_pickle(cv, path_in, name_in)
    return xform_data_t

vec_data = vec_fun(the_data.body, out_path, "vec", 1, 1, the_data.label)


'''
refactor vec_fun, allow user to switch between CV and TF-IDF 
'''

from sklearn.feature_extraction.text import CountVectorizer
    
from sklearn.feature_extraction.text import TfidfVectorizer



def vec_fun(df_in, path_in, name_in, m_in, n_in, label_in):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    import pandas as pd 
    
    if name_in == "tfidf": 
        cv = TfidfVectorizer(ngram_range=(m_in,n_in))
    elif name_in == "vec":
        cv = CountVectorizer(ngram_range=(m_in,n_in))
    else:
        print("Pick Either tfidf or vec")
        
    xform_data_t = pd.DataFrame(cv.fit_transform(df_in).toarray())
    xform_data_t.columns = cv.get_feature_names_out()
    xform_data_t.index = label_in
    
    # Save it in my out_path! 
    write_pickle(cv, path_in, name_in)
    return xform_data_t
    
vec_data = vec_fun(the_data.body, out_path, "tfidf", 1, 1, the_data.label)
    

    
vec_data, model_fun = extract_embeddings_pre(
    the_data.body, out_path, 'models/word2vec_sample/pruned.word2vec.txt')



model_fun.similar_by_word("fishing")

model_fun.similar_by_word("math")

king = model_fun.get_vector("king")
man = model_fun.get_vector("man")
woman = model_fun.get_vector("woman")
the_ans = king - man + woman # Queen 
the_tokens = model_fun.most_similar(the_ans)
 


# Create our own vectorized model (domain-specific)
vec_data, model_t = domain_train(the_data.body, out_path, None)
model_t.wv.most_similar("fishing")
model_t.wv.most_similar("math")





















