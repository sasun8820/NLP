#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:06:22 2023

@author: jeonseo
"""

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


vec_data = vec_fun(the_data.body_sw, out_path, "tfidf", 1, 1, the_data.label)

model_fun(vec_data, the_data.label, "rf", 0.20, out_path)



def grid_fun(df_in, lab_in, ts, grid_in, cv_i, name_in, o_path):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    
    
    X_train, X_test, y_train, y_test = train_test_split(df_in, lab_in,
                                                         test_size = ts,
                                                         random_state = 42)
     
    grid = {"n_estimators": [10,100], "max_depth": [None, 1]}
    
    model = RandomForestClassifier(random_state = 123)
    cv = GridSearchCV(model, grid_in, cv=cv_i)
    cv.fit(X_train, y_train)
    
    print("best perf", cv.best_score_)
    print("best params", cv.best_params_)
    opt_params = cv.best_params_
    
    '''
    Now using the best parameters, we train the entire model again with those parameters!
    '''
    model = RandomForestClassifier(**cv.best_params_, random_state=123)
    
    X_train, X_test, y_train, y_test = train_test_split(vec_data, the_data.label,
                                                         test_size = ts,
                                                         random_state = 42)
     
    model.fit(X_train, y_train)
    write_pickle(model, o_path, name_in)
    y_pred = model.predict(X_test)
    
    
    metrics = pd.DataFrame(precision_recall_fscore_support(y_test, y_pred, average="weighted"))
    metrics.index = ["precision", "recall", "F1", None]
    return model









