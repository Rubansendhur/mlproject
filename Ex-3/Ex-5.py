import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
drug_data = pd.read_csv("E:\cit-22smcb0055\SEM-V\Machine Learning\mlproject\Ex-3\drug200.csv")

# Encode categorical data
sex_encoder = LabelEncoder()
bp_encoder = LabelEncoder()
chol_encoder = LabelEncoder()

drug_data['Sex_encode'] = sex_encoder.fit_transform(drug_data['Sex'])
drug_data['BP_encode'] = bp_encoder.fit_transform(drug_data['BP'])
drug_data['Chol_encode'] = chol_encoder.fit_transform(drug_data['Cholesterol'])

# Feature and target variables
x = drug_data[['Age', 'Sex_encode', 'BP_encode', 'Chol_encode', 'Na_to_K']]
y = drug_data['Drug']

# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

scaler_minmax = MinMaxScaler()
x_minmax = scaler_minmax.fit_transform(x)

# Split dataset
x_train_scaled, x_test_scaled, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
x_train_minmax, x_test_minmax, _, _ = train_test_split(x_minmax, y, test_size=0.2, random_state=42)

# Define models
models = {
    'GaussianNB': GaussianNB(),
    'BernoulliNB': BernoulliNB()
}

# Add KNN with GridSearchCV for hyperparameter tuning
param_grid = {
    'n_neighbors': range(5, 20),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid_search.fit(x_train_scaled, y_train)
best_knn = grid_search.best_estimator_

# Print the best KNN parameters
print(f"Best KNN Parameters: {grid_search.best_params_}")
print(f"Best number of neighbors (k) for KNN: {grid_search.best_params_['n_neighbors']}")

models['KNN'] = best_knn

# Add MultinomialNB with MinMaxScaler
models['MultinomialNB'] = MultinomialNB()

# Train models and evaluate
def evaluate_models_with_scalers(models, x_train_std, x_test_std, x_train_minmax, x_test_minmax, y_train, y_test):
    results = {}
    for name, model in models.items():
        if name == 'MultinomialNB':
            x_train = x_train_minmax
            x_test = x_test_minmax
        else:
            x_train = x_train_std
            x_test = x_test_std
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        print(f"{name} Accuracy:", accuracy)
        print(f"{name} Classification Report:")
        print(results[name]['classification_report'])
        print(f"{name} Confusion Matrix:")
        print(results[name]['confusion_matrix'])
    return results

results = evaluate_models_with_scalers(models, x_train_scaled, x_test_scaled, x_train_minmax, x_test_minmax, y_train, y_test)

# Plot feature distributions
def plot_feature_distributions(df, features):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=20)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

plot_feature_distributions(drug_data, ['Age', 'Sex_encode', 'BP_encode', 'Chol_encode', 'Na_to_K'])

# Plot the accuracy scores of the models
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
sns.barplot(x=model_names, y=accuracies, palette='viridis')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.show()

# Plot confusion matrices
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

for name, result in results.items():
    plot_confusion_matrix(result['confusion_matrix'], f'{name} Confusion Matrix')

# Perform k-fold cross-validation for all models except MultinomialNB
kf = KFold(n_splits=10, shuffle=True, random_state=42)

def cross_val_evaluation(models, x, y, x_minmax):
    cv_results = {}
    for name, model in models.items():
        if name == 'MultinomialNB':
            cv_scores = cross_val_score(model, x_minmax, y, cv=kf, scoring='accuracy')
        else:
            cv_scores = cross_val_score(model, x, y, cv=kf, scoring='accuracy')
        cv_results[name] = {
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores)
        }
        print(f"{name} Cross-Validation Accuracy: Mean={cv_results[name]['mean_accuracy']:.4f}, Std={cv_results[name]['std_accuracy']:.4f}")
    return cv_results

cv_results = cross_val_evaluation(models, x_scaled, y, x_minmax)

# Plot k-fold cross-validation results
plt.figure(figsize=(10, 6))
cv_model_names = list(cv_results.keys())
cv_accuracies = [cv_results[name]['mean_accuracy'] for name in cv_model_names]
cv_std = [cv_results[name]['std_accuracy'] for name in cv_model_names]
sns.barplot(x=cv_model_names, y=cv_accuracies, palette='viridis', ci='sd')
plt.xlabel('Model')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Model Cross-Validation Accuracy Comparison')
plt.ylim(0, 1)
plt.show()

# Get unique values for each feature
unique_values = {
    'Sex': list(drug_data['Sex'].unique()),
    'BP': list(drug_data['BP'].unique()),
    'Cholesterol': list(drug_data['Cholesterol'].unique())
}

# Define a function to get input from the user and predict the drug
def get_user_input(prompt, options):
    while True:
        user_input = input(prompt).strip()
        if user_input in options:
            return user_input
        print(f"Invalid input. Please choose from {options}.")

age = float(input("Enter Age: "))
sex = get_user_input("Enter Sex (Male/Female): ", ['Male', 'Female'])
bp = get_user_input("Enter Blood Pressure (LOW/NORMAL/HIGH): ", ['LOW', 'NORMAL', 'HIGH'])
chol = get_user_input("Enter Cholesterol (NORMAL/HIGH): ", ['NORMAL', 'HIGH'])
na_to_k = float(input("Enter Na to K ratio: "))

# Map 'Male' and 'Female' to 'M' and 'F'
sex = 'M' if sex == 'Male' else 'F'

# Encode the user inputs
sex_encoded = sex_encoder.transform([sex])[0]
bp_encoded = bp_encoder.transform([bp])[0]
chol_encoded = chol_encoder.transform([chol])[0]

# Scale the input features
user_features = [[age, sex_encoded, bp_encoded, chol_encoded, na_to_k]]
user_features_scaled = scaler.transform(user_features)

# Predict the drug using the best KNN model
predicted_drug = best_knn.predict(user_features_scaled)[0]
print(f"The predicted drug is: {predicted_drug}")
