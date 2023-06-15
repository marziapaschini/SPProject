#!/usr/bin/env python
# coding: utf-8

# ### Data processing
# This file contains a comparison of several **supervised machine learning models** applied to pre-processed data.

# In[2]:


import data_preprocessing as dp
import numpy as np
import pandas as pd
import warnings
import pickle


# In[31]:


from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    chi2
) 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    train_test_split,
    cross_val_score,
    learning_curve,
    validation_curve
)
from sklearn.linear_model import (
    LogisticRegression,
    Ridge
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    f1_score
)
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint


# Split of the dataset in target and input variables

# In[4]:


y = dp.X['RainTomorrow']
X = dp.X.drop(['RainTomorrow'], axis=1)


# Function to split of the dataset in 2 parts (80%-20%), which is used in some models.

# In[10]:


def split_df(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    return X_train, X_test, y_train, y_test


# In[28]:


#X_train, X_test, y_train, y_test = split_df(X, y)


# #### Logistic Regression

# In[12]:


def logistic_regression(X_train, X_test, y_train, y_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression()
        model.fit(X_train, y_train)
    return model


# In[13]:


""" log_reg = logistic_regression(X_train, X_test, y_train, y_test)
accuracy_log_reg = log_reg.score(X_test, y_test)
print(accuracy_log_reg) """


# K-Fold Cross Validation

# In[43]:


""" def logistic_regression_kfold(X, y, n_splits=5, max_iter=800):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = LogisticRegression(max_iter=max_iter)
        scores = cross_val_score(model, X, y, cv=kfold)
        mean_score = np.mean(scores)
    return scores, mean_score
 """

# In[44]:


""" log_reg_scores, log_reg_mean_score = logistic_regression_kfold(X, y)
print("Cross-validation scores:", log_reg_scores)
print("Mean cross-validation score:", log_reg_mean_score)
 """

# #### Decision Tree

# In[13]:

""" 
def decision_tree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    return dt
 """

# In[14]:


""" dt = decision_tree(X_train, X_test, y_train, y_test)
accuracy_dt = dt.score(X_test, y_test)
print(accuracy_dt)

 """
# K-Fold Cross Validation

# In[45]:

""" 
def decision_tree_kfold(X, y, n_splits=5, max_iter=800):
    dt = DecisionTreeClassifier()
    cv_scores = cross_val_score(dt, X, y, cv=5)
    cv_scores_mean = cv_scores.mean()
    return cv_scores, cv_scores_mean
 """

# In[46]:

""" 
dt_scores, dt_mean_score = decision_tree_kfold(X, y)
print("Cross-validation scores:", dt_scores)
print("Mean cross-validation score:", dt_mean_score)
 """

# #### Random Forest

# In[54]:


def random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100) # 100 trees
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return rf, y_pred


# In[9]:


""" rf, y_pred = random_forest(X_train, X_test, y_train, y_test)
accuracy_rf = accuracy_score(y_test, y_pred)
print(accuracy_rf)

 """
# K-Fold Cross Validation

# In[51]:


""" def random_forest_kfold(X, y, n_splits=5):
    rf = RandomForestClassifier(n_estimators=100) # 100 trees
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=kfold)
    cv_scores_mean = cv_scores.mean()
    return cv_scores, cv_scores_mean
 """

# In[52]:


""" rf_cv_scores, rf_cv_scores_mean = random_forest_kfold(X, y)
print("CV scores:", rf_cv_scores)
print("CV mean: ", rf_cv_scores_mean)
 """

# #### SVM
# Linear, polinomial and radial basis function kernels.

# In[22]:
""" 

def svm_linear(X_train, X_test, y_train, y_test):
    svm_linear_model = svm.SVC(kernel='linear', C=1, random_state=42)
    svm_linear_model.fit(X_train, y_train)
    y_pred = svm_linear_model.predict(X_test)
    return svm_linear_model, y_pred


# In[27]:


def svm_poly(X_train, X_test, y_train, y_test):
    svm_poly_model = svm.SVC(kernel='poly', C=1, random_state=42)
    svm_poly_model.fit(X_train, y_train)
    y_pred = svm_poly_model.predict(X_test)
    return svm_poly_model, y_pred


# In[28]:


def svm_rbf(X_train, X_test, y_train, y_test):
    svm_rbf_model = svm.SVC(kernel='rbf', C=1, random_state=42)
    svm_rbf_model.fit(X_train, y_train)
    y_pred = svm_rbf_model.predict(X_test)
    return svm_rbf_model, y_pred


# In[24]:


svm_linear_model, y_pred = svm_linear(X_train, X_test, y_train, y_test)
accuracy_svm_linear_model = accuracy_score(y_test, y_pred)
print(accuracy_svm_linear_model)


# In[29]:


svm_poly_model, y_pred = svm_poly(X_train, X_test, y_train, y_test)
accuracy_svm_poly_model = accuracy_score(y_test, y_pred)
print(accuracy_svm_poly_model)


# In[30]:


svm_rbf_model, y_pred = svm_rbf(X_train, X_test, y_train, y_test)
accuracy_svm_rbf_model = accuracy_score(y_test, y_pred)
print(accuracy_svm_rbf_model)


# K-Fold Cross Validation

# In[55]:


def svm_kfold(X, y, n_splits=5):
    svm_model = svm.SVC(kernel='linear', C=1, random_state=42)
    svm_model.fit(X, y)
    cv_scores = cross_val_score(svm_model, X, y, cv=5)
    cv_scores_mean = cv_scores.mean()
    return cv_scores, cv_scores_mean


# In[57]:


svml_linear_cv_scores, svm_linear_cv_scores_mean = svm_kfold(X, y)
print("CV scores:", svml_linear_cv_scores)
print("Mean CV score:", svm_linear_cv_scores_mean)


# #### Accuracy comparison

# In[58]:


# create an accuracy dataframe
accuracies = pd.DataFrame({
    'Model': ['Logistic Regression', 'Logistic Regression KFold', 'Decision Tree', 'Decision Tree KFold', 'Random Forest', 'Random Forest KFold', 'SVM (Linear Kernel)', 'SVM (Linear Kernel) KFold', 'SVM (Polynomial Kernel)', 'SVM (RBF Kernel)'],
    'Accuracy': [accuracy_log_reg, log_reg_mean_score, accuracy_dt, dt_mean_score, accuracy_rf, rf_cv_scores_mean, accuracy_svm_linear_model, svm_linear_cv_scores_mean, accuracy_svm_rbf_model, accuracy_svm_poly_model]
})
print(accuracies)
# find the best accuracy
accuracies_sorted = accuracies.sort_values(by='Accuracy', ascending=False)
best_model = accuracies_sorted.iloc[0]['Model']
best_accuracy = accuracies_sorted.iloc[0]['Accuracy']


# In[59]:


print("Best model: ", best_model)
print("Accuracy: ", best_accuracy)

 """
# #### Random Forest Improvement

# In[65]:

""" 
param_distributions = {"n_estimators": randint(100, 1000),
                       "min_samples_split": randint(2, 20),
                       "min_samples_leaf": randint(1, 10),
                       "max_features": randint(1, 10)}

rf = RandomForestClassifier(max_depth=20)

random_search = RandomizedSearchCV(rf, param_distributions=param_distributions, n_iter=50, cv=5, n_jobs=-1)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_

rf_impr = RandomForestClassifier(n_estimators=best_params["n_estimators"],
                                  max_depth=20,
                                  min_samples_split=best_params["min_samples_split"],
                                  min_samples_leaf=best_params["min_samples_leaf"],
                                  max_features=best_params["max_features"])

rf_impr.fit(X_train, y_train)


# In[ ]:


accuracy_rf_impr = accuracy_score(y_test, y_pred_rf_impr)
print(accuracy_rf_impr)


# In[14]:


with open('log_reg.pkl', 'wb') as f:
    pickle.dump(log_reg, f)


# SelectKBest

# In[5]:


k = 20
selector = SelectKBest(score_func=f_classif, k=k)
X_new = selector.fit_transform(X, y)


# In[8]:


model = LogisticRegression(max_iter=1000)
model.fit(X_new, y)


# In[15]:


X_test_new = selector.transform(X_test)


# In[16]:


y_pred = model.predict(X_test_new)
y_prob = model.predict_proba(X_test_new)


# In[17]:


cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)


# In[20]:


# Calcola l'accuratezza del modello
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calcola la precisione del modello
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# Calcola il recall del modello
recall = recall_score(y_test, y_pred)
print('Recall:', recall)

# Calcola l'F1-score del modello
f1 = f1_score(y_test, y_pred)
print('F1-score:', f1)


# In[25]:


# Ottieni un array booleano che indica quali colonne sono state selezionate
selected_cols = selector.get_support()

# Elenca i nomi delle colonne selezionate
selected_features = X.columns[selected_cols]

# Stampa i nomi delle colonne selezionate
print(selected_features)


# In[32]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

new_selector = SelectKBest(chi2, k=14)
X_train_new2 = new_selector.fit_transform(X_train_scaled, y_train)
X_test_new2 = new_selector.transform(X_test_scaled)


# In[33]:


clf = LogisticRegression()
clf.fit(X_train_new2, y_train)
y_pred = clf.predict(X_test_new2)
accuracy = clf.score(X_test_new2, y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[36]:


# Ottieni un array booleano che indica quali colonne sono state selezionate
selected_cols = new_selector.get_support()

# Elenca i nomi delle colonne selezionate
new_selected_features = X.columns[selected_cols]

# Stampa i nomi delle colonne selezionate
print(new_selected_features)


# In[37]:


print(X.columns)

 """
# ### Additional columns work

# In[49]:


XXX = X.loc[:, ['MinTemp', 'MaxTemp', 'Rainfall', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm']]


# In[43]:


print(XXX.columns)


# In[50]:


XXX["Humidity"] = pd.concat([XXX["Humidity9am"], XXX["Humidity3pm"]], axis=1).mean(axis=1)
XXX = XXX.drop(['Humidity9am', 'Humidity3pm'], axis=1)
XXX["WindSpeed"] = pd.concat([XXX["WindSpeed9am"], XXX["WindSpeed3pm"]], axis=1).mean(axis=1)
XXX = XXX.drop(['WindSpeed9am', 'WindSpeed3pm'], axis=1)
print(XXX.columns)


# In[51]:


XXX_train, XXX_test, yyy_train, yyy_test = split_df(XXX, y)


# In[52]:


model = LogisticRegression(max_iter=1000)
model.fit(XXX_train, yyy_train)
y_pred = model.predict(XXX_test)


# In[53]:


# Calcola l'accuratezza del modello
accuracy = accuracy_score(yyy_test, y_pred)
print('Accuracy:', accuracy)

# Calcola la precisione del modello
precision = precision_score(yyy_test, y_pred)
print('Precision:', precision)

# Calcola il recall del modello
recall = recall_score(yyy_test, y_pred)
print('Recall:', recall)

# Calcola l'F1-score del modello
f1 = f1_score(yyy_test, y_pred)
print('F1-score:', f1)


# In[55]:


model_rf, y_pred_rf = random_forest(XXX_train, XXX_test, yyy_train, yyy_test)


# In[56]:


# Calcola l'accuratezza del modello
accuracy = accuracy_score(yyy_test, y_pred_rf)
print('Accuracy:', accuracy)

# Calcola la precisione del modello
precision = precision_score(yyy_test, y_pred_rf)
print('Precision:', precision)

# Calcola il recall del modello
recall = recall_score(yyy_test, y_pred_rf)
print('Recall:', recall)

# Calcola l'F1-score del modello
f1 = f1_score(yyy_test, y_pred_rf)
print('F1-score:', f1)

