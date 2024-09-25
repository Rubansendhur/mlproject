import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
import re

# Set pandas option to display 89 rows
pd.set_option('display.max_rows', 89)

# Load the dataset
data = pd.read_csv("C:\\Users\\ruban\\Downloads\\fifa clusterring\\kl.csv", encoding='ISO-8859-1')

# Display the first 5 rows
print("Initial Data:")
print(data.head())

# Check missing values in each column
print("\nMissing Values per Column:")
print(data.isnull().sum())

# Filter rows that have more than 50 missing values
data_with_many_missing = data[data.isnull().sum(axis=1) > 50]
print("\nRows with More Than 50 Missing Values:")
print(data_with_many_missing)

# Before deletion
print(f'\nBefore Deletion: {data.shape[0]} rows')

# Remove rows with more than 50 missing values
data = data[data.isnull().sum(axis=1) < 50]

# After deletion
print(f'After Deletion: {data.shape[0]} rows')

# Function to clean and convert currency columns
def clean_currency(value):
    try:
        value = re.sub(r'[^\d.]+', '', str(value))  # Keep only digits and dots
        return float(value)
    except ValueError:
        return np.nan  # Return NaN if conversion fails

# Selecting relevant columns for clustering
clustering_columns = ['Overall', 'Potential', 'Age', 'Value', 'Wage', 'Release Clause']
available_clustering_columns = [col for col in clustering_columns if col in data.columns]
clustering_data = data[available_clustering_columns].copy()

# Clean the 'Value', 'Wage', and 'Release Clause' columns
currency_columns = ['Value', 'Wage', 'Release Clause']
for col in currency_columns:
    if col in clustering_data.columns:
        clustering_data[col] = clustering_data[col].apply(clean_currency)
        print(f"Cleaned and converted '{col}' to numeric.")

# Drop rows with NaN values after cleaning
clustering_data = clustering_data.dropna()

# Rescaling the attributes
scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

# K-Means Clustering
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50, random_state=42)
    kmeans.fit(clustering_data_scaled)
    ssd.append(kmeans.inertia_)

# Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range_n_clusters, ssd, marker='o')
plt.title('Elbow Curve for K-Means')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances (Inertia)')
plt.grid(True)
plt.show()

# Silhouette analysis
print("\nSilhouette Scores for Different Cluster Numbers:")
silhouette_scores = {}
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50, random_state=42)
    cluster_labels = kmeans.fit_predict(clustering_data_scaled)
    silhouette_avg = silhouette_score(clustering_data_scaled, cluster_labels)
    silhouette_scores[num_clusters] = silhouette_avg
    print(f"For n_clusters = {num_clusters}, the silhouette score is {silhouette_avg:.4f}")

# Final K-Means model with k=3 (can be changed based on silhouette score or elbow method)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, max_iter=50, random_state=42)
clustering_labels = kmeans.fit_predict(clustering_data_scaled)
clustering_data['Cluster_Id'] = clustering_labels

# Merge cluster labels back to the original data
data_with_clusters = data.copy()
data_with_clusters['Cluster_Id'] = np.nan
data_with_clusters.loc[clustering_data.index, 'Cluster_Id'] = clustering_labels

# Visualizing K-Means Clusters using Scatter Plot (for 'Overall' vs 'Potential')
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Overall', y='Potential', hue='Cluster_Id', data=clustering_data, palette='Set1')
plt.title('K-Means Clustering: Overall vs Potential')
plt.xlabel('Overall')
plt.ylabel('Potential')
plt.show()

# Box plots for K-Means clusters
plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster_Id', y='Overall', data=data_with_clusters)
plt.title('Overall vs Cluster_Id (K-Means)')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster_Id', y='Potential', data=data_with_clusters)
plt.title('Potential vs Cluster_Id (K-Means)')
plt.show()

# Hierarchical Clustering
mergings = linkage(clustering_data_scaled, method="complete", metric='euclidean')
plt.figure(figsize=(15, 7))
dendrogram(mergings, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

# Cutting the dendrogram into 3 clusters
hierarchical_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
clustering_data['Hierarchical_Cluster'] = hierarchical_labels

# Assign hierarchical cluster labels back to the original data
data_with_clusters['Hierarchical_Cluster'] = np.nan
data_with_clusters.loc[clustering_data.index, 'Hierarchical_Cluster'] = hierarchical_labels

# Box plots for Hierarchical Clustering
plt.figure(figsize=(12, 6))
sns.boxplot(x='Hierarchical_Cluster', y='Overall', data=data_with_clusters)
plt.title('Overall vs Hierarchical_Cluster (Hierarchical Clustering)')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Hierarchical_Cluster', y='Potential', data=data_with_clusters)
plt.title('Potential vs Hierarchical_Cluster (Hierarchical Clustering)')
plt.show()

# Save the cleaned and segmented data
data_with_clusters.to_csv('fifa_segmented_players.csv', index=False)
print("\nSegmented data saved to 'fifa_segmented_players.csv'.")
