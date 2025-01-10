import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(data_url)

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

print(f"\nNumerical columns: {list(numerical_cols)}")
print(f"Categorical columns: {list(categorical_cols)}")

for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

for col in categorical_cols:
    if df[col].nunique() <= 20:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=df[col])
        plt.title(f'Count of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

if len(numerical_cols) > 1:
    correlation_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

if len(numerical_cols) > 1:
    sns.pairplot(df[numerical_cols])
    plt.title("Pair Plot")
    plt.show()

for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()

print("\nHandling Missing Values...")
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)
        print(f"Filled missing values in {col} with mean.")

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df.dropna(subset=[col], inplace=True)
        print(f"Dropped rows with missing values in {col}.")

print("\nCleaned Dataset Info:")
print(df.info())
