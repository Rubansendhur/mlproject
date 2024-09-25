import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Load the data into a Pandas DataFrame
data = pd.read_csv('E:\\cit-22smcb0055\\SEM-V\\Machine Learning\\mlproject\\Ex-3\\drug200.csv')

# Define the features and target variable
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']

# Convert categorical variables to numerical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])
X['BP'] = le.fit_transform(X['BP'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameter tuning spaces for DecisionTreeClassifier and RandomForestClassifier
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

param_grid_rf = {
    'n_estimators': [10, 50, 100, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

# Define the models
clf_dt = DecisionTreeClassifier(random_state=42)
clf_rf = RandomForestClassifier(random_state=42)

# Perform hyperparameter tuning using GridSearchCV for DecisionTreeClassifier and RandomizedSearchCV for
RandomForestClassifier
grid_search_dt = GridSearchCV(clf_dt, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
random_search_rf = RandomizedSearchCV(clf_rf, param_grid_rf, cv=5, scoring='accuracy', n_iter=10, random_state=42)

grid_search_dt.fit(X_train_scaled, y_train)
random_search_rf.fit(X_train_scaled, y_train)

# Get the best-performing models
best_clf_dt = grid_search_dt.best_estimator_
best_clf_rf = random_search_rf.best_estimator_

# Make predictions on the testing data using the best-performing models
y_pred_dt = best_clf_dt.predict(X_test_scaled)
y_pred_rf = best_clf_rf.predict(X_test_scaled)

# Evaluate the models
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print('Decision Tree Model:')
print('Accuracy:', accuracy_dt)
print('Classification Report:')
print(classification_report(y_test, y_pred_dt))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_dt))

print('\nRandom Forest Model:')
print('Accuracy:', accuracy_rf)
print('Classification Report:')
print(classification_report(y_test, y_pred_rf))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_rf))