import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the data
df = pd.read_csv("E:\cit-22smcb0055\SEM-V\Machine Learning\mlproject\Ex-4\employee_attrition_data.csv")

# Data preprocessing
df = df.drop_duplicates()
df = df.dropna()
df['Gender'] = df['Gender'].map({"Male": 1, "Female": 0})
df['Department'] = df['Department'].map({"Engineering": 0, "Finance": 1, "HR": 2, "Marketing": 3, "Sales": 4})
df['Job_Title'] = df['Job_Title'].map({"Engineer": 0, "Analyst": 1, "HR Specialist": 2, "Accountant": 3, "Manager": 4})

# Drop 'Employee_ID' if it's present
if 'Employee_ID' in df.columns:
    df = df.drop('Employee_ID', axis=1)
X = df.drop(columns=['Attrition'])
y = df['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for Decision Tree
param_grid = {
    'max_depth': [5, 10, 15, 320, 25, None],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': [None, 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Accuracy:", best_score)

# Train the Decision Tree with the best parameters
clf = DecisionTreeClassifier(**best_params, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
report = classification_report(y_test, y_pred, target_names=["No Attrition", "Attrition"])
print("Classification Report:")
print(report)

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# Now apply Bagging with the tuned Decision Tree
bagging_clf = BaggingClassifier(estimator=clf, n_estimators=100, random_state=42, n_jobs=-1)
bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_test)

# Evaluate Bagging Classifier with Decision Tree
print("\nBagging with Decision Tree Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_bagging):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_bagging))
print("Classification Report:")
print(classification_report(y_test, y_pred_bagging))

# Predefined user input for prediction
user_input = {
    'Age': 35,
    'Gender': 1,  # Male
    'Department': 0,  # Engineering
    'Job_Title': 0,  # Engineer
    'Years_at_Company': 5,
    'Satisfaction_Level': 0.8,
    'Average_Monthly_Hours': 160,
    'Promotion_Last_5Years': 0,  # No
    'Salary': 70000
}

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])

# Ensure user_df has the same columns as X
user_df = user_df.reindex(columns=X.columns, fill_value=0)

# Predict based on user input with consistent column names
user_prediction = clf.predict(user_df)
print("Decision Tree Prediction:", "Attrition" if user_prediction[0] == 1 else "No Attrition")

user_prediction_bagging = bagging_clf.predict(user_df)
print("Bagging Classifier Prediction:", "Attrition" if user_prediction_bagging[0] == 1 else "No Attrition")
