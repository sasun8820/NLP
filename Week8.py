#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:01:49 2023

@author: jeonseo
"""

import os
os.chdir('/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/')

from Utils import *

data_path = "/Users/jeonseo/Desktop/QMSS/Fall 2023/NLP/Natural-Language-Processing-NLP-/data/"

the_data = file_walker(data_path)

#  Vectorization: Transforming text-based data into numeric values 

# apply(function) to certain column
the_data['body_sw'] = the_data.body.apply(rem_sw)

the_data['body_cnt'] = the_data.body.apply(token_counts)
the_data['body_sw_cnt'] = the_data.body_sw.apply(token_counts)



the_data['body_cnt_u'] = the_data.body.apply(token_counts_unique)
the_data['body_sw_cnt_u'] = the_data.body_sw.apply(token_counts_unique)


the_data["body_sw_stem"] = the_data.body_sw.apply(stem_fun)
the_data["body_sw_stem_cnt"] = the_data.body_sw_stem.apply(token_counts)
the_data["body_sw_stem_cnt_u"] = the_data.body_sw_stem.apply(token_counts_unique)

vec_data = vec_fun(the_data.body_sw, out_path, "tfidf", 1, 3, the_data.label)
    
# vec_data, model_fun = extract_embeddings_pre(
#     the_data.body, out_path, 'models/word2vec_sample/pruned.word2vec.txt')


# vec_data, model_t = domain_train(the_data.body, out_path, None)


'''
Cosine Similarity: Measure similarity between two vectors by caclculating the cosine of the angle 
between the two vectors. If it's more similar, the angle is smaller, and vice versa.
'''


def cos_fun(df_in_a, df_in_b, label_in):
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    
    cos_sim = pd.DataFrame(cosine_similarity(df_in_a, df_in_b))
    cos_sim.index = label_in
    cos_sim.columns = label_in
    
    return cos_sim

cos_sim_final = cos_fun(vec_data, vec_data, the_data.label)



vec_data_chi = chi_fun(vec_data, the_data.label, 100, out_path, "chi")


'''
Principal Component Analysis (PCA): Does feature engineering process 
- Decouple collinearity by reducing dimensions? 
- n_components: 정수이면 total variance 상관없이 그냥 10개의 components로 나누라는 말이 됨
- n_components: float (0.08)으로 쓰면 total variance를 8%로 정할테니, 몇개의 components를 줄지 보여줌
- 보통 total variance가 낮으면, n_components가 낮음 --> 음.. 그냥 반대로 선택지가 많으면 큰 variance를
가질 수 있으니까.

'''

from sklearn.decomposition import PCA
 # If I want only 0.8% of variance, how many components would I need?
pca_fun = PCA(n_components = 0.08) # components 40개 나옴 
# pca_fun = PCA(n_components = 10) We got 0.098 % of varainces 
pca_data = pca_fun.fit_transform(vec_data)

# Decides the componenets by the standard deviation
# first principal component explains the maximum variance in the data, 
# the second principal component explains the second most variance, and so on...
exp_var = sum(pca_fun.explained_variance_ratio_)


'''
create a function called pca_fun
enable user to select explained variance
save off as pickle object
'''

def pca_fun(df_in, exp_var, path_in, name_in):
    from sklearn.decomposition import PCA
    pca_fun = PCA(n_components = exp_var)
    pca_data = pca_fun.fit_transform(df_in)
    exp_var = sum(pca_fun.explained_variance_ratio_)
    write_pickle(pca_fun, path_in, "pca")
    return pca_data 


pca_d_fun = pca_fun(vec_data, 0.95, out_path, "pca")

pca_d_fun_2 = pca_fun(vec_data, 0.8, out_path, "pca")






















