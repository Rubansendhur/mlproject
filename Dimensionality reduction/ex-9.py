import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.feature_selection import chi2, RFE, SelectKBest, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# Load and prepare data
data = pd.read_csv("C:\\Users\\ruban\\Downloads\\ml\\data.csv")
data = data.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')  # Handle missing columns gracefully
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
target = data['diagnosis']

# Use MinMaxScaler to ensure the data is non-negative
features = data.drop('diagnosis', axis=1)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_data = pd.DataFrame(scaled_features, columns=features.columns)
scaled_data['diagnosis'] = target

# Plot the correlation matrix
correlation = scaled_data.corr()
plt.figure(figsize=(20, 15))
plt.title('Correlation Between Predictor Variables')
sns.heatmap(correlation, annot=False, fmt='.2f', cmap='coolwarm')
plt.show()

# Train-test split
X = scaled_data.drop('diagnosis', axis=1)
y = scaled_data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Function to evaluate the model
def evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # F1 Score
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.show()

    return accuracy

# Evaluate without feature selection
print("Initial Random Forest Evaluation")
baseline_accuracy = evaluate_model(X_train, X_test, y_train, y_test, rf_model)

# Feature Selection Techniques
## 1. Filtering Method: Chi-Squared Test
print("Feature Selection using Chi-Squared Test")
chi2_selector = SelectKBest(chi2, k=5)  # Select the top 5 features
X_train_chi2 = chi2_selector.fit_transform(X_train, y_train)
X_test_chi2 = chi2_selector.transform(X_test)
chi2_accuracy = evaluate_model(X_train_chi2, X_test_chi2, y_train, y_test, rf_model)

# Replace the previous Variance Threshold method evaluation with this
print(f"Chi-Squared Accuracy: {chi2_accuracy:.4f}")

## 2. Filtering Method: Mutual Information
print("Feature Selection using Mutual Information")
mi_selector = SelectKBest(mutual_info_classif, k=10)
X_train_mi = mi_selector.fit_transform(X_train, y_train)
X_test_mi = mi_selector.transform(X_test)
mutual_info_accuracy = evaluate_model(X_train_mi, X_test_mi, y_train, y_test, rf_model)

## 3. Wrapper Method: Recursive Feature Elimination (RFE)
print("Feature Selection using RFE")
rfe_selector = RFE(estimator=rf_model, n_features_to_select=10, step=1)
X_train_rfe = rfe_selector.fit_transform(X_train, y_train)
X_test_rfe = rfe_selector.transform(X_test)
rfe_accuracy = evaluate_model(X_train_rfe, X_test_rfe, y_train, y_test, rf_model)

## 4. Embedded Method: Random Forest Feature Importance
print("Feature Selection using Random Forest Importance")
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:] 
X_train_emb = X_train.iloc[:, indices]
X_test_emb = X_test.iloc[:, indices]
rf_importance_accuracy = evaluate_model(X_train_emb, X_test_emb, y_train, y_test, rf_model)

print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

# Collect and print the best method
accuracies = {
    "Chi-Squared": chi2_accuracy,
    "Mutual Information": mutual_info_accuracy,
    "RFE": rfe_accuracy,
    "Random Forest Importance": rf_importance_accuracy
}


best_method = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_method]
print(f"\nBest Method: {best_method} with accuracy of {best_accuracy:.4f}")
