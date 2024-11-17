import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Step 1: Load the dataset from the CSV file
file_path = "Iris.csv"  
data = pd.read_csv(file_path)

# Assuming the last column is the target (species) and the rest are features
X = data.iloc[:, :-1].values  # Features: Sepal and Petal measurements
true_labels = data.iloc[:, -1].values  # True species labels

# Convert species names to numeric values for compatibility
species_to_num = {species: idx for idx, species in enumerate(set(true_labels))}
num_to_species = {idx: species for species, idx in species_to_num.items()}
true_labels_numeric = [species_to_num[label] for label in true_labels]

# Step 2: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means Clustering
n_clusters = 3  # Number of clusters (assuming 3 species)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# Cluster labels assigned by K-Means
cluster_labels = kmeans.labels_

# Step 4: Map Cluster Labels to True Labels
# Use the mode of the true numeric labels in each cluster
label_mapping = {}
for cluster in range(n_clusters):
    mask = cluster_labels == cluster
    cluster_true_labels = [true_labels_numeric[i] for i in range(len(cluster_labels)) if mask[i]]
    if cluster_true_labels:  # Ensure the cluster is not empty
        label_mapping[cluster] = mode(cluster_true_labels, keepdims=True).mode[0]

# Map the predicted cluster labels to their corresponding true numeric labels
predicted_labels_numeric = [label_mapping[cluster] for cluster in cluster_labels]

# Convert predicted numeric labels back to species names
predicted_labels = [num_to_species[label] for label in predicted_labels_numeric]

# Step 5: Evaluate the Clustering with Accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"K-Means Accuracy: {accuracy:.2f}")

# Step 6: Optional - Print Cluster to Species Mapping
mapped_species = {cluster: num_to_species[label] for cluster, label in label_mapping.items()}
print("Cluster-to-Species Mapping:", mapped_species)