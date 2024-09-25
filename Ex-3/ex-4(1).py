import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
drug_data = pd.read_csv("D:\\cit-22smcb0055\\SEM-V\\Machine Learning\\mlproject\\Ex-3\\drug200.csv")

# Encode categorical data
sex_encoder = LabelEncoder()
bp_encoder = LabelEncoder()
chol_encoder = LabelEncoder()

drug_data['Sex_encode'] = sex_encoder.fit_transform(drug_data['Sex'])
drug_data['BP_encode'] = bp_encoder.fit_transform(drug_data['BP'])
drug_data['Chol_encode'] = chol_encoder.fit_transform(drug_data['Cholesterol'])

# Add the target variable to the correlation matrix
drug_data['Drug_encode'] = LabelEncoder().fit_transform(drug_data['Drug'])

# Calculate correlation matrix
correlation_matrix = drug_data[['Age', 'Sex_encode', 'BP_encode', 'Chol_encode', 'Na_to_K', 'Drug_encode']].corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

# Identify the feature with the highest correlation with the target variable
target_correlation = correlation_matrix['Drug_encode'].drop('Drug_encode').abs()
highest_correlation_feature = target_correlation.idxmax()
print(f"Feature with the highest correlation with the target variable: {highest_correlation_feature} ({target_correlation.max()})")

# Select features with low correlation to the highest correlation feature
selected_features = [highest_correlation_feature]
threshold = 0.1  # Define a threshold for low correlation
for feature in correlation_matrix.columns.drop(['Drug_encode', highest_correlation_feature]):
    if abs(correlation_matrix[highest_correlation_feature][feature]) < threshold:
        selected_features.append(feature)

print(f"Selected features: {selected_features}")

# Prepare the selected features and target variable
x = drug_data[selected_features]
y = drug_data['Drug']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define models
models = {
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB()
}

# Train models and evaluate
def evaluate_models(models, x_train, y_train, x_test, y_test):
    results = {}
    for name, model in models.items():
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

results = evaluate_models(models, x_train, y_train, x_test, y_test)

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

# Perform k-fold cross-validation
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=42)

def cross_val_evaluation(models, x, y):
    cv_results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, x, y, cv=kf, scoring='accuracy')
        cv_results[name] = {
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores)
        }
        print(f"{name} Cross-Validation Accuracy: Mean={cv_results[name]['mean_accuracy']:.4f}, Std={cv_results[name]['std_accuracy']:.4f}")
    return cv_results

cv_results = cross_val_evaluation(models, x, y)

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

# Ask for input from the user and predict the drug
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

# Encode user inputs using respective encoders
sex_encode = sex_encoder.transform([sex])[0]
bp_encode = bp_encoder.transform([bp])[0]
chol_encode = chol_encoder.transform([chol])[0]

user_data = np.array([[age, sex_encode, bp_encode, chol_encode, na_to_k]])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Predict with the best model (Gaussian Naive Bayes in this case)
    user_pred = models['GaussianNB'].predict(user_data)

print(f"The predicted drug is: {user_pred[0]}")
