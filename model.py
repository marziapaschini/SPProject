#!/usr/bin/env python
# coding: utf-8

# In[15]:


import data_preprocessing as dp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# In[3]:


y = dp.X['RainTomorrow']
X = dp.X.drop(['RainTomorrow'], axis=1)


# In[4]:


def split_df(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    return X_train, X_test, y_train, y_test


# In[5]:


def random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100) # 100 trees
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return rf, y_pred


# In[6]:


X_train, X_test, y_train, y_test = split_df(X, y)


# In[7]:


model, y_pred = random_forest(X_train, X_test, y_train, y_test)
