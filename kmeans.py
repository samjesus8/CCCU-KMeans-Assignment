import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import multiprocessing

def mainProgram():
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

def elbowMethod():
    # Load the Iris dataset
    iris_df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    X = iris_df.iloc[:, :-1].values

    # Calculate distortion score for different values of K
    distortions = []
    K = range(1, 11) # This will do it from 1-11 clusters
    for k in K:
        kmeans = KMeans(n_clusters=k) # Execute K means with this value of K and get the result
        kmeans.fit(X)
        distortions.append(kmeans.inertia_) # The inertia property returns how well the data was clustered

    # Plot the distortion score for different values of K
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Distortion score')
    plt.title('Elbow Method')
    plt.show()

# We want to show both graphs at once, so we use threading to run both methods at once
# This if statement makes sure that new processes are made ONLY after the main process is done initializing
if __name__ == "__main__":
    # First, Declare the processes we will be running
    process1 = multiprocessing.Process(target=mainProgram)
    process2 = multiprocessing.Process(target=elbowMethod)

    # Start these processes
    process1.start()
    process2.start()

    # Finally, join them together to run simultaneously
    process1.join()
    process2.join()
