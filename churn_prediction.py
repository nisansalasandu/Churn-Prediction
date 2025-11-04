# Level 3 - Task 1

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('churn-bigml-20.csv')

# View dataset information
print(df.info())

# Display first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# No any missing values found in the dataset

# Check for duplicate rows
print(f"Number of duplicate rows: {df.duplicated().sum()}")

# No any duplicate rows found in the dataset

# Data Preprocessing

# Display column names
print(df.columns)

# Drop irrelevant columns
df = df.drop(columns=['State', 'Area code'], errors='ignore')

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col]) 
    
# Check dataset after encoding
print(df.head())

# Separate features(X) and target variable(Y)
X = df.drop('Churn', axis=1)
Y = df['Churn']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Model Training and Evaluation
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    print(f"\nModel: {model_name}")
    print("Accuracy:", accuracy_score(Y_test, Y_pred))
    print("Precision:", precision_score(Y_test, Y_pred))
    print("Recall:", recall_score(Y_test, Y_pred))
    print("F1 Score:", f1_score(Y_test, Y_pred))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
    print()
    
    # The best model based on accuracy, precision, recall, and F1 score can be selected for further tuning or deployment.
    # Choose the best model as Random Forest for hyperparameter tuning
best_model = RandomForestClassifier()
best_model.fit(X_train, Y_train)
Y_pred = best_model.predict(X_test)
print("\nBest Model: Random Forest")

# Confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')    

# Save the plot
plt.savefig('confusion_matrix_random_forest.png')
plt.show()

# Classification Report
print("Classification Report for Random Forest:\n", classification_report(Y_test, Y_pred))

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, Y_train)

print("Best Hyperparameters:", grid.best_params_)
print("Best Accuracy", grid.best_score_)

# Final evaluation with the best model
final_model = grid.best_estimator_
Y_final_pred = final_model.predict(X_test)
print("\nFinal Model Evaluation after Hyperparameter Tuning:")
print("Accuracy:", accuracy_score(Y_test, Y_final_pred))


    