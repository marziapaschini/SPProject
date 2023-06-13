import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    KFold,
    GridSearchCV,
    train_test_split,
    cross_val_score,
    learning_curve,
    validation_curve
)
from sklearn.preprocessing import StandardScaler
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
    roc_auc_score
)

# Load data
df_unclean = pd.read_csv('weather.csv')

# DATA CLEANING & REDUCTION
# 1. REMOVE USELESS COLUMNS
def remove_columns(df, columns):
    return df.drop(columns=columns, inplace=False)
df = remove_columns(df_unclean, ["Evaporation", "Sunshine", "RISK_MM"])

# 2. FILLING IN MISSING VALUES
# fill by mean values (grouping by location)
def fill_missing_values_with_mean(df, columns, groupby):
    for col in columns:
        df[col] = df.groupby(groupby)[col].transform(lambda x: x.fillna(x.mean()))
    return df
df = fill_missing_values_with_mean(df, ["MinTemp", "MaxTemp", "Rainfall", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Temp9am", "Temp3pm"], "Location")

# fill by most common value (grouping by location only in some cases)
def fill_missing_values_with_mode(df, columns, groupby=None):
    for col in columns:
        if groupby is not None:
            df[col] = df.groupby(groupby)[col].transform(lambda x: x.fillna(x.mode()[0]))
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df
df = fill_missing_values_with_mode(df, ["WindDir9am", "WindDir3pm"], "Location")
df = fill_missing_values_with_mode(df, ["Cloud9am", "Cloud3pm", "WindGustSpeed", "Pressure9am", "Pressure3pm", "WindGustDir"])

# remove some rows because of no "RainToday" values
def remove_rows_with_missing_values(df, subset):
    return df.dropna(subset=subset, inplace=False)
df = remove_rows_with_missing_values(df, ["RainToday"])

# missing values check
def check_missing_values(df):
    if df.isnull().sum().any():
        return false
    else:
        return "Null values: " + str(df.isnull().sum())
#print(check_missing_values(df))

# 3. IDENTIFYING & SMOOTHING OUTLIERS
# numeric columns boxplot => "Rainfall" columns has outliers
numeric_cols = df.select_dtypes(include=[np.number])
def plot_numeric_columns_boxplot(df):
    fig, ax = plt.subplots()
    ax.boxplot(numeric_cols.values)
    ax.set_title('Numeric columns boxplot')
    ax.set_xticklabels(numeric_cols.columns, rotation=45)
    ax.set_ylabel('Values')
    plt.show()
#plot_numeric_columns_boxplot(df)

def replace_outliers_with_mobile_mean(df, column_name, window_size, std_dev):
    # compute mean mobile value
    mean_mobile = df[column_name].rolling(window_size).mean()
    # replace outliers with mean mobile value
    outlier_threshold = std_dev * df[column_name].std()
    outliers = abs(df[column_name] - df[column_name].shift()) > outlier_threshold
    df.loc[outliers, column_name] = mean_mobile[outliers]
    return df
df = replace_outliers_with_mobile_mean(df, 'Rainfall', 3, 2)

# DATA TRANSFORMATION
# 1. DATA TYPES CONVERSION
def data_types_conversion(df, date_columns=[], bool_columns=[]):
    if date_columns:
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
    if bool_columns:
        for col in bool_columns:
            df[col] = df[col].map({'No': False, 'Yes': True})
    return df
df = data_types_conversion(df, date_columns=['Date'], bool_columns=['RainToday', 'RainTomorrow'])

#print(df.iloc[:10, [1, 5]])
#print("Columns types: " + str(df.dtypes) + "\n")
#print(df['RainTomorrows'].unique())

# NaN values check
def nan_values_check(df):
    num_nan = df.isnull().sum()
    return num_nan
#nan_counts = count_nan_values(df)
#print("NaN: ", nan_counts)

#print(len(df)) #24721 entries
#print(df.info()) # 22 columns

# 2. DUMMIES CONVERSION
def dummies_processing(df):
    cat_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    X_cat = df[cat_cols]
    X_cat = pd.get_dummies(X_cat, drop_first=True)
    X_cat['RainToday'] = df['RainToday']
    X_cat['RainTomorrow'] = df['RainTomorrow']
    return X_cat

# final preprocessing step
X_numeric = df[['MinTemp', 'MaxTemp', 'Rainfall', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Temp9am', 'Temp3pm', 'Pressure9am', 'Pressure3pm']]
X_categoric = dummies_processing(df)
#print(X_numeric.columns)

# scale numeric data: zero mean & unitary standard deviation
def scale_numeric_data(df, numeric_cols):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numeric_cols])
    df[numeric_cols] = df_scaled
    return df
#X_numeric = scale_numeric_data(df, X_numeric.columns)

# combine (standardized) numerical and categorical features
#X = np.concatenate((X_numeric, X_categoric), axis=1) # no metadata, just 2 matrices
X = pd.concat([X_numeric, X_categoric], axis=1)

# PREDICTIVE MODEL
# 1. DIVIDE DATASET IN INPUT AND TARGET VARIABLES
y = df['RainTomorrow']
#X = X.drop(['Unnamed: 0', 'Date', 'Location', 'RainTomorrow', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1)
X = X.drop(['RainTomorrow'], axis=1)

# SPLIT DATASET IN 80-20
def split_df(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = split_df(X, y)

# 2a. LOGISTIC REGRESSION
def logistic_regression_accuracy(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy
#print(logistic_regression_accuracy(X_train, X_test, y_train, y_test))

 # 2b. LOGISTIC REGRESSION WITH CROSS VALIDATION
# K-fold cross-validation with 5 folds
def logistic_regression_kfold(X, y, n_splits=5, max_iter=800):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=max_iter)
    scores = cross_val_score(model, X, y, cv=kfold)
    mean_score = np.mean(scores)
    return scores, mean_score
# scores, mean_score = logistic_regression_kfold(X, y)
# print("Cross-validation scores:", scores)
# print("Mean cross-validation score:", mean_score)


# 3a. DECISION TREE
def decision_tree_accuracy(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    accuracy = dt.score(X_test, y_test)
    return accuracy
#print("Decision tree accuracy", decision_tree_accuracy(X_train, X_test, y_train, y_test))


# 3b. DECISION TREE WITH CROSS VALIDATION
def decision_tree_kfold(X, y, n_splits=5, max_iter=800):
    dt = DecisionTreeClassifier()
    cv_scores = cross_val_score(dt, X, y, cv=5)
    cv_scores_mean = cv_scores.mean()
    return cv_scores, cv_scores_mean
# cv_scores, cv_scores_mean = decision_tree_kfold(X, y)
# print("CV scores:", cv_scores)
# print("CV mean: ", cv_scores_mean)


# 4a. RANDOM FOREST
def random_forest_accuracy(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100) # 100 trees
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
#print("Random forest accuracy:", random_forest_accuracy(X_train, X_test, y_train, y_test))


# 4b. RANDOM FOREST WITH CROSS VALIDATION
def random_forest_kfold(X, y, n_splits=5):
    rf = RandomForestClassifier(n_estimators=100) # 100 trees
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=kfold)
    cv_scores_mean = cv_scores.mean()
    return cv_scores, cv_scores_mean
# cv_scores, cv_scores_mean = random_forest_kfold(X, y)
# print("CV scores:", cv_scores)
# print("CV mean: ", cv_scores_mean)


# 5a. SVM
def svm_accuracy(X_train, X_test, y_train, y_test):
    svm_model = svm.SVC(kernel='linear', C=1, random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
#print("SVM accuracy:", svm_accuracy(X_train, X_test, y_train, y_test))


# 5a. SVM WITH CROSS VALIDATION
def svm_kfold(X, y, n_splits=5):
    svm_model = svm.SVC(kernel='linear', C=1, random_state=42)
    svm_model.fit(X, y)
    cv_scores = cross_val_score(svm_model, X, y, cv=5)
    cv_scores_mean = cv_scores.mean()
    return cv_scores, cv_scores_mean
# cv_scores, cv_scores_mean = svm_kfold(X, y)
# print("CV scores:", cv_scores)
# print("Mean CV score:", cv_scores.mean())


# EVALUATION METRICS
# confusion matrix
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# precision
def calculate_precision(model, X_test, y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1])
    return precision

# precision = calculate_precision(model, X_test, y_test)
# print("Precision: ", precision)

# recall
def calculate_recall(model, X_test, y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    recall = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
    return recall

# recall = calculate_recall(model, X_test, y_test)
# print("Recall: ", recall)

# AUC-ROC
def calculate_auc_roc(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:,1]
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    return auc_roc

# model = LogisticRegression()
# model.fit(X_train, y_train)

# auc_roc = calculate_auc_roc(model, X_test_scaled, y_test)
# print("AUC-ROC: ", auc_roc)

""" # learning curve
train_sizes, train_scores, test_scores = learning_curve(
    estimator=model, X=X, y=y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_squared_error'
)

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, test_scores_mean, label='Validation error')
plt.xlabel('Training set size')
plt.ylabel('Error')
plt.legend()
plt.show() """
