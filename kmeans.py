import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
# Extracting the data from this data-set
X = iris_df.iloc[:, :-1].values

# Executing the built-in K-Means algorithm
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Getting the cluster centers and labels to add onto the graph
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the data points and cluster centers
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[labels == 2, 0], X[labels == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(centers[:, 0], centers[:,1], s = 200, marker='x', linewidths=3, color='black', label = 'Centroids')

#Display the legend
plt.legend()

#Finally show the output
plt.show()
