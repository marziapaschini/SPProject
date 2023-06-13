import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv('weather.csv')

# columns
numeric_cols = df.select_dtypes(include=[np.number])
numeric_cols = numeric_cols.drop(numeric_cols.columns[0], axis=1)
categoric_cols = df.select_dtypes(include='object').columns.tolist()
categoric_cols = pd.Index(categoric_cols).delete([0, 1])

print(numeric_cols.columns)
print(categoric_cols)

# city names
city_names = df["Location"].unique()
def fix_city_names(names):
    fixed_city_names = []
    for name in names:
        # applica la regola di inserire uno spazio tra una lettera minuscola e una lettera maiuscola consecutive in una parola in camel case
        name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
        # converte il nome in formato corretto
        name = ' '.join(word.capitalize() for word in name.split(' '))
        fixed_city_names.append(name)
    return fixed_city_names
fixed_city_names = fix_city_names(city_names)
#print("Cities", fixed_city_names)

# data distribution boxplot
def plot_numeric_distribution(df, numeric_cols):
    sns.boxplot(data=numeric_cols)
    plt.xticks(rotation=45)
    return plt.gcf()
def plot_categoric_distribution(df, categoric_cols):
    for col in categoric_cols:
        plt.figure(figsize=(10, 8))
        sns.countplot(x=col, data=df)
        plt.title(f'{col} distribution')
        return plt.gcf()

# missing numerical data
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
    return plt.gcf()

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
    return plt.gcf()

# correlation analysis
def plot_numeric_correlation(df, numeric_cols):
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation matrix")
    plt.xticks(rotation=45)
    return plt.gcf()

# distribution analysis
def plot_temps_distribution(df):
    plt.figure(figsize=(10, 8))
    sns.histplot(df['MinTemp'], kde=True)
    sns.kdeplot(df['MinTemp'], color='red')
    plt.title('Data distribution')
    return plt.gcf()

# numerical distribution analysis by city
# specific column
def plot_column_numeric_distribution_by_city(df, column):
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='Location', y=column, data=df)
    plt.title(f'{column} distribution by city')
    plt.xticks(rotation=15)
    return plt.gcf()
# all columns
def plot_numeric_distribution_by_city(df, numeric_cols):
    plots = []
    for col in numeric_cols:
        plt.figure(figsize=(10, 8))
        sns.boxplot(x='Location', y=col, data=df)
        plt.title(f'{col} distribution by city')
        plt.xticks(rotation=15)
        plot = plt.gcf()
        plots.append(plot)
    return plots

# categoric distribution analysis by city
# specific column
def plot_column_categoric_distribution_by_city(df, column):
    grafico = df.groupby([column, 'Location']).size().unstack().plot(kind='bar', stacked=True)
    plt.title(f'{column} distribution by city')
    plt.xlabel(column)
    plt.ylabel("Occurrences number")
    return plt.gcf()
# all columns
def plot_categoric_distribution_by_city(df, categoric_cols):
    plots = []
    for col in categoric_cols:
        sns.catplot(x=col, kind="count", hue="Location", data=df, height=4, aspect=2)
        plt.title(f'{col} distribution by city')
        plt.xlabel(col)
        plt.ylabel("Occurrences number")
        plt.xticks(rotation=15)
        plot = plt.gcf()
        plots.append(plot)
    return plots