#!/usr/bin/env python
# coding: utf-8

# In[76]:


import nltk
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer


# # Data

# In[76]:


def features(sentence, index):
    return {
        'word': sentence[index],
        'is_capitalized': sentence[index].capitalize() == sentence[index],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][:1] if len(sentence[index]) >= 1 else 'SHORT',
        'prefix-2': sentence[index][:2] if len(sentence[index]) >= 2 else 'SHORT',
        'prefix-3': sentence[index][:3] if len(sentence[index]) >= 3 else 'SHORT',
        'suffix-1': sentence[index][-1:] if len(sentence[index]) >= 1 else 'SHORT',
        'suffix-2': sentence[index][-2:] if len(sentence[index]) >= 2 else 'SHORT',
        'suffix-3': sentence[index][-3:] if len(sentence[index]) >= 3 else 'SHORT',
        'prev_word': 'START' if index == 0 else sentence[index - 1],
        'next_word': 'END' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:],
    }


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged_sent in tagged_sentences:
        untagged_sent = untag(tagged_sent)
        for index in range(len(tagged_sent)):
            X.append(features(untagged_sent, index))
            y.append(tagged_sent[index][1])

    return X, y

# get data
tagged_sentences = nltk.corpus.treebank.tagged_sents()
X, y = transform_to_dataset(tagged_sentences)

label_encoder = LabelEncoder().fit(list(set(y)))
y = label_encoder.transform(y)


# # Target Algorithm Estimate

# In[79]:


# Target Algorithm Estimate
# This function accepts a configuration space point
# and returns a number (the score of the provided config)
def my_tae(cfg):
    
    cfg = dict(cfg)
    
    # ML pipeline
    clf = Pipeline(
        [
            ('vectorizer', DictVectorizer(sparse=False)),
            ('classifier', LogisticRegression(**cfg)),
        ]
    )

    # TODO this should probably be outside the function
    f1_scorer = make_scorer(f1_score, average='weighted')
################### USE FOR CROSS-VAL ##########################
#     # k-fold cross validation
#     # TODO as we only need one score, better to use cross_val_score
#     cv_results = cross_validate(
#         estimator=clf,
#         X=X_use,
#         y=y_use,
#         cv=3,
#         return_train_score=False,
# #        return_estimator=False,
#         scoring={'weighted_f1':f1_scorer},
#         verbose=5,
#     )absabs

#     score = cv_results['test_weighted_f1'].mean()

################# SINGLE TRAIN-TEST SPLIT ################################
    X_train, X_test, y_train, y_test = train_test_split(X_use, y_use, random_state=42)
    clf.fit(X_train, y_train)
    
    score =  f1_scorer(clf, X_test, y_test)

    return max(0, 1 - score)


# # Configuration Space

# In[80]:


# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario

# Define the Configuration space dimensions and limits
# N.B. 
random_state = Constant("random_state", 42)
solver = Constant("solver", "liblinear")
C = UniformFloatHyperparameter("C", 0.03, 10, default_value=1, log=True)

cs1 = ConfigurationSpace()

cs1.add_hyperparameter(random_state)
cs1.add_hyperparameter(solver)
cs1.add_hyperparameter(C)


# # Optimisation

# In[ ]:


from smac.facade.smac_facade import SMAC

nuse = 2500
X_use = X[:nuse]
y_use = y[:nuse]

# scenario dict
scenario_kwargs = {
    "run_obj": "quality",   # optimize quality (alternatively runtime)
    "runcount-limit": 50,  # maximum function evaluations
    "cs": cs1,               # configuration space
    "deterministic": "true",
    "output_dir": f"smac_output",
}

smac = SMAC(
    scenario=Scenario(scenario_kwargs),
    rng=np.random.RandomState(42),
    tae_runner=my_tae,
)

incumbent = smac.optimize()

