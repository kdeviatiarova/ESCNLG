import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from yellowbrick.cluster import KElbowVisualizer
import warnings
from contextlib import contextmanager

warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def suppress_plot():
    plt.ioff()
    yield
    plt.ion()
    plt.close('all')


def cluster_student_results(data, num_clusters_range=(4, 10)):
    """
    Cluster student test results using KMeans and Agglomerative clustering.
    This function performs min-max scaling, PCA, finds the optimal number of clusters,
    compares clustering algorithms, and returns a DataFrame with mean scores per cluster.

    Args:
    - data (pd.DataFrame): The input data containing student test results.
    - num_clusters_range (tuple): The range of cluster numbers to consider for clustering.

    Returns:
    - mean_scores_per_cluster (pd.DataFrame): DataFrame containing mean scores per cluster.
    """
    if 'index' in data.columns:
      data = data.drop('index', axis=1)

    # Min-Max Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # PCA
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)

    # Elbow Method for KMeans

    visualizer = KElbowVisualizer(KMeans(), k=(2, 10), timings=True)
    with suppress_plot():
        visualizer.fit(pca_data)
    optimal_num_clusters = visualizer.elbow_value_

    # Silhouette Method for Agglomerative Clustering

    silhouette_scores = []
    for i in range(num_clusters_range[0], num_clusters_range[1]):
        agglomerative = AgglomerativeClustering(n_clusters=i)
        labels = agglomerative.fit_predict(pca_data)
        silhouette_scores.append(silhouette_score(pca_data, labels))
    num_clusters_agglomerative = np.argmax(silhouette_scores) + num_clusters_range[0]

    # Reduce dimensionality for clustering visualization

    pca = PCA(n_components=2)
    X = pca.fit_transform(pca_data)

    # KMeans clustering

    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # Agglomerative clustering

    agglomerative = AgglomerativeClustering(n_clusters=num_clusters_agglomerative)
    agglomerative_labels = agglomerative.fit_predict(X)

    # Calculating metrics

    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    agglomerative_silhouette = silhouette_score(X, agglomerative_labels)

    kmeans_davies_bouldin = davies_bouldin_score(X, kmeans_labels)
    agglomerative_davies_bouldin = davies_bouldin_score(X, agglomerative_labels)

    kmeans_calinski_harabasz = calinski_harabasz_score(X, kmeans_labels)
    agglomerative_calinski_harabasz = calinski_harabasz_score(X, agglomerative_labels)

    kmeans_better = 0
    agglomerative_better = 0

    # Comparison of Metrics
    # Silhouette Score (higher is better)

    if kmeans_silhouette > agglomerative_silhouette:
        kmeans_better += 1
    else:
        agglomerative_better += 1

    # Davies-Bouldin Index (lower is better)

    if kmeans_davies_bouldin < agglomerative_davies_bouldin:
        kmeans_better += 1
    else:
        agglomerative_better += 1

    # Calinski-Harabasz Index (higher is better)

    if kmeans_calinski_harabasz > agglomerative_calinski_harabasz:
        kmeans_better += 1
    else:
        agglomerative_better += 1

    # Majority voting on three metrics

    if kmeans_better > agglomerative_better:
        best_labels = kmeans_labels
        best_algorithm = "KMeans"
    else:
        best_labels = agglomerative_labels
        best_algorithm = "Agglomerative"

    # Plot clusters

    plot_clusters(X, best_labels)

    # Calculate mean score per cluster

    df = pd.DataFrame(scaled_data, columns=data.columns)
    df['Cluster'] = best_labels + 1
    mean_scores_per_cluster = df.groupby('Cluster').mean()

    # Plot heatmap of mean scores per cluster

    plot_heatmap(mean_scores_per_cluster)

    print(f"Best clustering algorithm: {best_algorithm}")
    return mean_scores_per_cluster


def plot_elbow_method(data, num_clusters_range=(2, 10)):
    """
    Plot the Elbow Method to find the optimal number of clusters.

    Args:
    - distortions (list): List of distortion values for different cluster numbers.
    - num_clusters_range (tuple): Range of cluster numbers to consider.

    Returns:
    - int: Optimal number of clusters.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)

    visualizer = KElbowVisualizer(KMeans(), k=num_clusters_range, timings=False)
    visualizer.fit(pca_data)
    visualizer.show()
    return visualizer.elbow_value_


def plot_silhouette_method(data, num_clusters_range=(2, 10)):
    """
    Plot the Silhouette Method to find the optimal number of clusters.

    Args:
    - silhouette_scores (list): List of silhouette scores for different cluster numbers.
    - num_clusters_range (tuple): Range of cluster numbers to consider.

    Returns:
    - int: Optimal number of clusters based on the highest silhouette score.
    """
    max_silhouette_score = float('-inf')
    best_n_clusters = None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)

    silhouette_scores = []
    for n_clusters in range(num_clusters_range[0], num_clusters_range[1] + 1):

        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(pca_data)
        score = silhouette_score(pca_data, labels)
        silhouette_scores.append(score)

        if n_clusters!= 2 and n_clusters >= num_clusters_range[0] and score > max_silhouette_score:
            max_silhouette_score = score
            best_n_clusters = n_clusters

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_clusters_range[0], num_clusters_range[1] + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Best N of Clusters')

    if best_n_clusters:
        plt.axvline(x=best_n_clusters, color='r', linestyle='--', label=f'Best: {best_n_clusters} clusters')

    plt.legend()
    plt.show()


def plot_clusters(data, labels):
    """
    Plot the clustered data.

    Args:
    - data (np.ndarray): Data to be plotted.
    - labels (np.ndarray): Cluster labels for the data.
    """
    plt.scatter(data[:, 0], data[:, 1], c=labels + 1, cmap='plasma')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Clustered Student Results')
    plt.show()


def plot_heatmap(mean_scores_per_cluster):
    """
    Plot a heatmap of mean scores per cluster.

    Args:
    - mean_scores_per_cluster (pd.DataFrame): DataFrame containing mean scores per cluster.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(mean_scores_per_cluster, cmap='plasma', annot=True, fmt=".2f")
    plt.title('Normalized Mean Scores per Cluster')
    plt.xlabel('Features')
    plt.ylabel('Cluster')
    plt.show()