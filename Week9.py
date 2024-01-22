#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:01:58 2023

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


vec_data = vec_fun(the_data.body_sw, out_path, "tfidf", 1, 3, the_data.label)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd 

model = RandomForestClassifier(random_state=123, max_depth = 500)


X_train, X_test, y_train, y_test = train_test_split(vec_data, the_data.label,
                                                    test_size = 0.20,
                                                    random_state = 42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred_proba = model.predict_proba(X_test)
y_pred_proba = pd.DataFrame(model.predict_proba(X_test))
y_pred_proba.columns = model. classes_


metrics = pd.DataFrame(precision_recall_fscore_support(y_test, y_pred, average="weighted"))
metrics.index = ["precision", "recall", "F1", None]


fi = pd.DataFrame(model.feature_importances_)
fi.index = model.feature_names_in_
fi.columns = ["score"]

num_fi = fi[fi.score != 0]
len(num_fi) / len(fi) # only 1.9% are meaningful as their f1 score isn't 0



'''
Create a function called model_fun that takes in a dataframe 
and a label, user selects test/train size, function saves off 
the trained and return the model, prints the metrics, and saves off the
feature importance
'''


def model_fun(df_in, label_in, sel_in, t_in, o_in):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd 
    
    model = RandomForestClassifier(random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(df_in, label_in,
                                                        test_size = t_in,
                                                        random_state = 42)
    model.fit(X_train, y_train)
    write_pickle(model, o_in, sel_in)
    

    y_pred = model.predict(X_test)
    y_pred_proba = pd.DataFrame(model.predict_proba(X_test))
    y_pred_proba.columns = model.classes_
    metrics = pd.DataFrame(precision_recall_fscore_support(y_test, y_pred, average="weighted"))
    metrics.index = ["precision", "recall", "F1", None]
    
    fi = pd.DataFrame(model.feature_importances_)
    fi.index = model.feature_names_in_
    fi.columns = ["score"]
    num_fi = fi[fi.score != 0]

    fi.to_csv(o_in + "fi.csv", index=True)
    print(metrics)
    return model

    
model_fun(vec_data, the_data.label, "rf", 0.20, out_path)


'''
rework model_fun to allow user to select between random forest and 
gassuan naive bayes
'''

def model_fun(df_in, label_in, sel_in, t_in, o_in):
    
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd 
    
    if sel_in == "nb":
        model = GaussianNB()
    elif sel_in == "rf":
        model = RandomForestClassifier(random_state=123)
        
    X_train, X_test, y_train, y_test = train_test_split(df_in, label_in,
                                                        test_size = t_in,
                                                        random_state = 42)
    model.fit(X_train, y_train)
    write_pickle(model, o_in, sel_in)
    

    y_pred = model.predict(X_test)
    y_pred_proba = pd.DataFrame(model.predict_proba(X_test))
    y_pred_proba.columns = model.classes_
    
    try:

        fi = pd.DataFrame(model.feature_importances_)
        fi.index = model.feature_names_in_
        fi.columns = ["score"]
        num_fi = fi[fi.score != 0]
        fi.to_csv(o_in + "fi.csv", index=True)
    except:
        print("Issue occured")
    
    metrics = pd.DataFrame(precision_recall_fscore_support(y_test, y_pred, average="weighted"))
    metrics.index = ["precision", "recall", "F1", None]
    
    #print(metrics)

    return model


model_fun(vec_data, the_data.label, "nb", 0.20, out_path)


from sklearn.decomposition import PCA
 # If I want only 0.8% of variance, how many components would I need?
# pca_fun = PCA(n_components = 0.08) # components 40개 나옴 


pca_d_fun = pca_fun(vec_data, 0.95, out_path, "pca")


the_model_fun = model_fun(
    pca_d_fun, the_data.label, "rf", .20, out_path)







