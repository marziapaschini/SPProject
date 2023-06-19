import data_preprocessing as dp
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

y = dp.X['RainTomorrow']
X = dp.X.drop(['RainTomorrow'], axis=1)

def split_df(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    return X_train, X_test, y_train, y_test

def random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100) # 100 trees
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return rf, y_pred

X_train, X_test, y_train, y_test = split_df(X, y)

model, y_pred = random_forest(X_train, X_test, y_train, y_test)
