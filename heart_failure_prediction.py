# Heart Failure Prediction Analysis Script
# Author: MSc Dissertation Project (LD7083)
# Description: Full EDA and ML modeling using heart_failure_prediction.csv dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("heart_failure_prediction.csv")

# Display basic information
print("Dataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())
print("\nClass Balance:")
print(df['HeartDisease'].value_counts())

# Create output folder (optional)
import os
os.makedirs("plots", exist_ok=True)

# 1. Countplot for categorical variables
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in categorical_features:
    plt.figure()
    sns.countplot(x=col, hue='HeartDisease', data=df)
    plt.title(f"Heart Disease by {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/countplot_{col}.png")

# 2. Histogram for numerical variables
numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
for col in numerical_features:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"plots/histogram_{col}.png")

# 3. Boxplots of numerical features by target
for col in numerical_features:
    plt.figure()
    sns.boxplot(x='HeartDisease', y=col, data=df)
    plt.title(f"{col} vs Heart Disease")
    plt.tight_layout()
    plt.savefig(f"plots/boxplot_{col}.png")

# 4. Violin plots
for col in numerical_features:
    plt.figure()
    sns.violinplot(x='HeartDisease', y=col, data=df)
    plt.title(f"{col} (Violin) vs Heart Disease")
    plt.tight_layout()
    plt.savefig(f"plots/violinplot_{col}.png")

# 5. Pairplot
sns.pairplot(df, hue='HeartDisease', vars=numerical_features)
plt.savefig("plots/pairplot.png")

# 6. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")

# 7. Pie chart of target variable
plt.figure()
df['HeartDisease'].value_counts().plot.pie(autopct='%1.1f%%', labels=['No Disease', 'Disease'], colors=['lightblue', 'salmon'])
plt.title("Heart Disease Distribution")
plt.ylabel('')
plt.savefig("plots/piechart_target.png")

# Data Preprocessing
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression Model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Results")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Decision Tree Model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Results")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Feature Importance Plot (Decision Tree)
feature_names = X.columns
importances = pd.Series(dt.feature_importances_, index=feature_names)
plt.figure()
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances (Decision Tree)")
plt.tight_layout()
plt.savefig("plots/feature_importance.png")

print("\nEDA and Modeling complete. All plots saved to the 'plots' folder.")
