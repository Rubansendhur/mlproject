import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load dataset
drug_original = pd.read_csv("D:\\cit-22smcb0055\\SEM-V\\Machine Learning\\Ex-3\\drug200.csv")

# Encode categorical data
label = LabelEncoder()  

drug_original['Sex_encode'] = label.fit_transform(drug_original['Sex'])
drug_original['BP_encode'] = label.fit_transform(drug_original['BP'])
drug_original['Chol_encode'] = label.fit_transform(drug_original['Cholesterol'])

x = drug_original[['Age', 'Sex_encode', 'BP_encode', 'Chol_encode', 'Na_to_K']]
y = drug_original.Drug

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the model before tuning
dtc_before_tuning = DecisionTreeClassifier(max_depth=2,random_state=0)
dtc_before_tuning.fit(x_train, y_train)

# Predict and evaluate on test data before tuning
y_test_pred_before_tuning = dtc_before_tuning.predict(x_test)
print("Before Tuning:")
print(classification_report(y_test, y_test_pred_before_tuning))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_before_tuning))

# Define the parameter grid
param_grid = {'max_depth': np.arange(1, 20)}

# Perform grid search with cross-validation
grid_search = GridSearchCV(dtc_before_tuning, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Best parameters and best score
best_max_depth = grid_search.best_params_['max_depth']
best_score = grid_search.best_score_

print(f"Best max_depth: {best_max_depth}")
print(f"Best cross-validation score: {best_score}")

# Print cross-validation scores for all max_depth values
cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print(f"max_depth: {params['max_depth']} - Cross-validation score: {mean_score}")

# Train the final model with the best max_depth
dtc_after_tuning = DecisionTreeClassifier(criterion='gini', max_depth=best_max_depth, random_state=0)
dtc_after_tuning.fit(x_train, y_train)

# Predict and evaluate on test data after tuning
y_test_pred_after_tuning = dtc_after_tuning.predict(x_test)
print("After Tuning:")
print(classification_report(y_test, y_test_pred_after_tuning))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_after_tuning))

# Visualize the decision tree before tuning
plt.figure(figsize=(15,10))
plot_tree(dtc_before_tuning, feature_names=['Age','Sex_encode','BP_encode','Chol_encode','Na_to_K'], class_names=dtc_before_tuning.classes_, filled=True, fontsize=10)
plt.title("Decision Tree Before Tuning")
plt.show()

# Visualize the decision tree after tuning
plt.figure(figsize=(15,10))
plot_tree(dtc_after_tuning, feature_names=['Age','Sex_encode','BP_encode','Chol_encode','Na_to_K'], class_names=dtc_after_tuning.classes_, filled=True, fontsize=10)
plt.title("Decision Tree After Tuning")
plt.show()

# Visualize the cross-validation scores
plt.plot(np.arange(1, 20), cv_results['mean_test_score'], marker='o')
plt.xlabel('max_depth')
plt.xticks(np.arange(1, 20))
plt.ylabel('Cross-validation score')
plt.title('Cross-validation scores for different max_depth values')
plt.show()

# Print unique values for each encoded feature
print("Unique values for Sex:", drug_original['Sex'].unique())
print("Unique values for BP:", drug_original['BP'].unique())
print("Unique values for Cholesterol:", drug_original['Cholesterol'].unique())

# Ask for input from the user and predict the drug
age = float(input("Enter Age: "))
sex = input("Enter Sex (Male/Female): ")
bp = input("Enter Blood Pressure (LOW/NORMAL/HIGH): ")
chol = input("Enter Cholesterol (NORMAL/HIGH): ")
na_to_k = float(input("Enter Na to K ratio: "))

sex_encode = label.transform([sex])[0]
bp_encode = label.transform([bp])[0]
chol_encode = label.transform([chol])[0]

user_data = np.array([[age, sex_encode, bp_encode, chol_encode, na_to_k]])
user_pred = dtc_after_tuning.predict(user_data)
print(f"The predicted drug is: {user_pred[0]}")
