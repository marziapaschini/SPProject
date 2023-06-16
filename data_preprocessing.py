#!/usr/bin/env python
# coding: utf-8

# ### Data pre-processing
# In this file, the dataset is processed to make it suitable for the subsequent machine learning analysis.

# In[70]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[71]:


# load data
df = pd.read_csv('weather.csv')


# In[7]:


# dataframe column names
print(df.columns)


# In[72]:


# numeric and categoric columns
numeric_cols = df.select_dtypes(include=[np.number])
numeric_cols = numeric_cols.drop(numeric_cols.columns[0], axis=1)
categoric_cols = df.select_dtypes(include='object').columns.tolist()
categoric_cols = pd.Index(categoric_cols).delete([0, 1])


# Correlation matrix

# In[16]:


def plot_numeric_correlation(df, numeric_cols):
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation matrix")
    plt.xticks(rotation=45)
    plt.show()


# In[17]:


#df_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#df_numeric_cols.remove('Unnamed: 0')
#plot_numeric_correlation(df, df_numeric_cols)


# In[14]:


#print("Numeric columns: ", numeric_cols.columns)
#print("\nCategoric columns: ", categoric_cols)


# #### Missing and null data analysis

# In[73]:


# missing numeric data
def missing_numeric_data(df, numeric_cols):
    missing_data = df.isnull().sum()
    print("Missing data per column:")
    print(missing_data)
    plt.figure(figsize=(10, 5))
    plt.bar(missing_data.index, missing_data.values)
    plt.xticks(rotation=90)
    plt.xlabel("Columns")
    plt.ylabel("Missing data")
    plt.title("Missing data per column")
    plt.show()

# missing categoric data
def missing_categoric_data(df, categoric_cols):
    missing_data = df[categoric_cols].isnull().sum()
    print("Missing data per column:")
    print(missing_data)
    plt.figure(figsize=(10, 5))
    plt.bar(missing_data.index, missing_data.values)
    plt.xticks(rotation=90)
    plt.xlabel("Columns")
    plt.ylabel("Missing data")
    plt.title("Missing data per column")
    plt.xticks(rotation=15)
    plt.show()
    
def nan_values_check(df):
    num_nan = df.isnull().sum()
    return num_nan


# In[74]:


#missing_numeric_data(df, numeric_cols)


# In[75]:


#missing_categoric_data(df, categoric_cols)


# In[76]:


#print(nan_values_check(df))


# #### Data cleaning and reduction
# Functions to remove useless columns, fill by mean values, fill by most common value, remove rows with no prediction (i.e. "RainTomorrow" values).

# In[77]:


def remove_columns(df, columns):
    return df.drop(columns=columns, inplace=False)

def fill_missing_values_with_mean(df, columns, groupby):
    for col in columns:
        df[col] = df.groupby(groupby)[col].transform(lambda x: x.fillna(x.mean()))
    return df

def fill_missing_values_with_mode(df, columns, groupby=None):
    for col in columns:
        if groupby is not None:
            df[col] = df.groupby(groupby)[col].transform(lambda x: x.fillna(x.mode()[0]))
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def remove_rows_with_missing_values(df, subset):
    return df.dropna(subset=subset, inplace=False)

def check_missing_values(df):
    if df.isnull().sum().any():
        return false
    else:
        return "Null values: " + str(df.isnull().sum())


# In[78]:


df = remove_columns(df, ["Evaporation", "Sunshine", "RISK_MM"])
#print(df.columns)


# In[79]:


df = fill_missing_values_with_mean(df, ["MinTemp", "MaxTemp", "Rainfall", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Temp9am", "Temp3pm"], "Location")
df = fill_missing_values_with_mode(df, ["WindDir9am", "WindDir3pm"], "Location")
df = fill_missing_values_with_mode(df, ["Cloud9am", "Cloud3pm", "WindGustSpeed", "Pressure9am", "Pressure3pm", "WindGustDir"])
df = remove_rows_with_missing_values(df, ["RainToday"])

#print(check_missing_values(df))


# In[83]:


# update sub-dataframes
numeric_cols = df.select_dtypes(include=[np.number])
categoric_cols = df.select_dtypes(include='object').columns.tolist()


# Functions to identify and smoothing outliers.

# In[84]:


def plot_numeric_columns_boxplot(df):
    fig, ax = plt.subplots()
    ax.boxplot(numeric_cols.values)
    ax.set_title('Numeric columns boxplot')
    ax.set_xticklabels(numeric_cols.columns, rotation=45)
    ax.set_ylabel('Values')
    plt.show()
    
def replace_outliers_with_mobile_mean(df, column_name, window_size, std_dev):
    # compute mean mobile value
    mean_mobile = df[column_name].rolling(window_size).mean()
    # replace outliers with mean mobile value
    outlier_threshold = std_dev * df[column_name].std()
    outliers = abs(df[column_name] - df[column_name].shift()) > outlier_threshold
    df.loc[outliers, column_name] = mean_mobile[outliers]
    return df


# In[85]:


#plot_numeric_columns_boxplot(df)


# In[86]:


df = replace_outliers_with_mobile_mean(df, 'Rainfall', 3, 2)
plot_numeric_columns_boxplot(numeric_cols)


# #### Data transformation
# Functions to perform data types conversion and applying dummies to categoric variables.

# In[87]:


# 'Date': from object to date-time
# 'RainToday' and 'RainTomorrow': from 'Yes'/No' to 'True'/'False'
def data_types_conversion(df, date_columns=[], bool_columns=[]):
    if date_columns:
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
    if bool_columns:
        for col in bool_columns:
            df[col] = df[col].map({'No': False, 'Yes': True})
    return df

def dummies_processing(df):
    cat_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    X_cat = df[cat_cols]
    X_cat = pd.get_dummies(X_cat, drop_first=True)
    X_cat['RainToday'] = df['RainToday']
    X_cat['RainTomorrow'] = df['RainTomorrow']
    return X_cat


# In[88]:


df = data_types_conversion(df, date_columns=['Date'], bool_columns=['RainToday', 'RainTomorrow'])
#print("NaN: \n", nan_values_check(df))
X_categoric = dummies_processing(df)
X_numeric = df[['MinTemp', 'MaxTemp', 'Rainfall', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Temp9am', 'Temp3pm', 'Pressure9am', 'Pressure3pm']]
#print(nan_values_check(X_categoric))


# In[89]:


# pre-processed dataframe
X = pd.concat([X_numeric, X_categoric], axis=1)
#print(X.head())



# #### Columns selection
# For the final dataset, only the most important and user-friendly columns are kept.

# In[33]:


X = X.loc[:, ['MinTemp', 'MaxTemp', 'Rainfall', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'RainToday', 'RainTomorrow']]


# In[34]:


X["Humidity"] = pd.concat([X["Humidity9am"], X["Humidity3pm"]], axis=1).mean(axis=1)
X = X.drop(['Humidity9am', 'Humidity3pm'], axis=1)
X["WindSpeed"] = pd.concat([X["WindSpeed9am"], X["WindSpeed3pm"]], axis=1).mean(axis=1)
X = X.drop(['WindSpeed9am', 'WindSpeed3pm'], axis=1)


# In[35]:


print(X.columns)

