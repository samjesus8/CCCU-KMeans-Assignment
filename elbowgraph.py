import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
X = iris_df.iloc[:, :-1].values

# Calculate distortion score for different values of K
distortions = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# Plot the distortion score for different values of K
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Distortion score')
plt.title('Elbow Method')
plt.show()
