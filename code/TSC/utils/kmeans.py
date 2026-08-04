import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import NDArray
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def perform_kmeans_clustering(data: NDArray, n_clusters: int, random_state=42, plot_elbow=True):
    """
    Perform K-means clustering on the input data.

    Parameters:
    data (numpy.ndarray): Input data with shape (9996, 91)
    n_clusters (int): Number of clusters for K-means (default: 5)
    random_state (int): Random state for reproducibility (default: 42)
    plot_elbow (bool): Whether to plot the elbow curve (default: True)

    Returns:
    tuple: (cluster_labels, cluster_centers, fig1, fig2)
        - cluster_labels: Array of cluster labels for each data point
        - cluster_centers: Array of cluster centers
        - fig1: Figure object for the elbow curve (or None if plot_elbow is False)
        - fig2: Figure object for the PCA visualization of clusters
    """
    avg_jm = np.mean(data, axis=1)

    # Step 1: Preprocess the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Step 2: Determine the optimal number of clusters (optional)
    # if plot_elbow:
    #     inertias = []
    #     k_range = range(1, 11)
    #
    #     for k in k_range:
    #         kmeans = KMeans(n_clusters=k, random_state=random_state)
    #         kmeans.fit(data_scaled)
    #         inertias.append(kmeans.inertia_)
    #
    #     # Plot the elbow curve
    #     fig1, ax1 = plt.subplots()
    #     ax1.plot(k_range, inertias, 'bx-')
    #     ax1.set_xlabel('k')
    #     ax1.set_ylabel('Inertia')
    #     ax1.set_title('Elbow Method for Optimal k')
    # else:
    #     fig1 = None

    # Step 3: Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(data_scaled)
    u_labels = np.unique(labels)

    # This part selects the best feature in every cluster.
    selected_features = []
    for i in u_labels:
        arr = np.copy(avg_jm)
        indices_to_exclude = np.where(labels == i)
        value_to_set = -10
        mask = np.ones_like(arr, dtype=bool)
        mask[indices_to_exclude] = False
        arr[mask] = value_to_set
        selected_features.append(np.argmax(arr))

    return selected_features


# Example usage:
if __name__ == "__main__":
    # Generate sample data (replace this with your actual data)
    data = np.random.rand(9996, 91)

    # Perform clustering
    labels, centers, elbow_fig, cluster_fig = perform_kmeans_clustering(data, n_clusters=5)

    # Print results
    print("Cluster centers:")
    print(centers)

    print("\nCluster sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} data points")

    # Show figures
    plt.show()
