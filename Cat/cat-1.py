import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

data = pd.read_excel('E:/cit-22smcb0055/SEM-V/Machine Learning/mlproject/Cat/csv_result-MagicTelescope.xlsx')

#eda
print(data.head())
print(data.describe())
print(data.isnull().sum())
data.fillna(data.mean(numeric_only=True), inplace=True)

# Encode the target variable
le = LabelEncoder()
data['class'] = le.fit_transform(data['class']) 

# Separate features and target
X = data.drop(['ID', 'class'], axis=1)
y = data['class']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Calculate correlation matrix
# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Initialize the models
models = {
    'GaussianNB': GaussianNB(),
    'BernoulliNB': BernoulliNB()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        'Accuracy': accuracy,
        'Classification Report': report,
        'Confusion Matrix': cm
    }

# Print results
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {result['Accuracy']:.2f}")
    print(f"Classification Report:\n{result['Classification Report']}")

    print(f"Confusion Matrix:\n{result['Confusion Matrix']}")
    

    plt.figure(figsize=(6, 4))
    sns.heatmap(result['Confusion Matrix'], annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix ({name})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

cv_results_before_tuning = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    cv_results_before_tuning[name] = cv_scores.mean() * 100

print("Cross-Validation Accuracy (Before Tuning):")
for name, accuracy in cv_results_before_tuning.items():
    print(f"{name}: {accuracy:.2f}%")

tuned_models = {
    'GaussianNB': GaussianNB(),
    'BernoulliNB': BernoulliNB(alpha=0.5) 
}

cv_results_after_tuning = {}
for name, model in tuned_models.items():
    cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    cv_results_after_tuning[name] = cv_scores.mean() * 100

print("Cross-Validation Accuracy (After Tuning):")
for name, accuracy in cv_results_after_tuning.items():
    print(f"{name}: {accuracy:.2f}%")

before_tuning_acc = list(cv_results_before_tuning.values())
after_tuning_acc = list(cv_results_after_tuning.values())
model_names = list(cv_results_before_tuning.keys())

best_model_name = max(cv_results_after_tuning, key=cv_results_after_tuning.get)
print(f"Best Model After Tuning: {best_model_name} with accuracy: {cv_results_after_tuning[best_model_name]:.2f}%")


