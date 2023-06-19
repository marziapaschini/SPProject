import numpy as np
import pandas as pd

# load data
df = pd.read_csv('weather.csv')

# numeric and categoric columns
numeric_cols = df.select_dtypes(include=[np.number])
numeric_cols = numeric_cols.drop(numeric_cols.columns[0], axis=1)
categoric_cols = df.select_dtypes(include='object').columns.tolist()
categoric_cols = pd.Index(categoric_cols).delete([0, 1])

# #### Missing and null data analysis

def nan_values_check(df):
    num_nan = df.isnull().sum()
    return num_nan

# #### Data cleaning and reduction
# Functions to remove useless columns, fill by mean values, fill by most common value, remove rows with no prediction (i.e. "RainTomorrow" values).

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
        return False
    else:
        return "Null values: " + str(df.isnull().sum())

df = fill_missing_values_with_mean(df, ["MinTemp", "MaxTemp", "Rainfall", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Temp9am", "Temp3pm"], "Location")
df = fill_missing_values_with_mode(df, ["WindDir9am", "WindDir3pm"], "Location")
df = fill_missing_values_with_mode(df, ["Cloud9am", "Cloud3pm", "WindGustSpeed", "Pressure9am", "Pressure3pm", "WindGustDir"])
df = remove_rows_with_missing_values(df, ["RainToday"])

# update sub-dataframes
numeric_cols = df.select_dtypes(include=[np.number])
categoric_cols = df.select_dtypes(include='object').columns.tolist()

# Functions to identify and smoothing outliers.
  
def replace_outliers_with_mobile_mean(df, column_name, window_size, std_dev):
    # compute mean mobile value
    mean_mobile = df[column_name].rolling(window_size).mean()
    # replace outliers with mean mobile value
    outlier_threshold = std_dev * df[column_name].std()
    outliers = abs(df[column_name] - df[column_name].shift()) > outlier_threshold
    df.loc[outliers, column_name] = mean_mobile[outliers]
    return df

df = replace_outliers_with_mobile_mean(df, 'Rainfall', 3, 2)

# #### Data transformation
# Functions to perform data types conversion and applying dummies to categoric variables.

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

df = data_types_conversion(df, date_columns=['Date'], bool_columns=['RainToday', 'RainTomorrow'])
X_categoric = dummies_processing(df)
X_numeric = df[['MinTemp', 'MaxTemp', 'Rainfall', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Temp9am', 'Temp3pm', 'Pressure9am', 'Pressure3pm']]

# pre-processed dataframe
X = pd.concat([X_numeric, X_categoric], axis=1)

X = X.loc[:, ['MinTemp', 'MaxTemp', 'Rainfall', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'RainToday', 'RainTomorrow']]

X["Humidity"] = pd.concat([X["Humidity9am"], X["Humidity3pm"]], axis=1).mean(axis=1)
X = X.drop(['Humidity9am', 'Humidity3pm'], axis=1)
X["WindSpeed"] = pd.concat([X["WindSpeed9am"], X["WindSpeed3pm"]], axis=1).mean(axis=1)
X = X.drop(['WindSpeed9am', 'WindSpeed3pm'], axis=1)
