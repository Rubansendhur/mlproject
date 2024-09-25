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
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# Load and prepare data
data = pd.read_csv("Dimensionality reduction\data.csv")
print(data.head())
data = data.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')  # Handle missing columns gracefully
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
target = data['diagnosis']

# Use MinMaxScaler to ensure the data is non-negative
features = data.drop('diagnosis', axis=1)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_data = pd.DataFrame(scaled_features, columns=features.columns)
scaled_data['diagnosis'] = target

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


# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)
print("Random Forest Evaluation with PCA")
pca_accuracy = evaluate_model(X_train_pca, X_test_pca, y_train_pca, y_test_pca, rf_model)

# Factor Analysis
fa = FactorAnalysis(n_components=2)
X_fa = fa.fit_transform(X)
X_train_fa, X_test_fa, y_train_fa, y_test_fa = train_test_split(X_fa, y, test_size=0.3, random_state=42)
print("Random Forest Evaluation with Factor Analysis")
fa_accuracy = evaluate_model(X_train_fa, X_test_fa, y_train_fa, y_test_fa, rf_model)

