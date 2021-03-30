import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# loading dataset
digits = load_digits()

data, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size

# printing number of digits samples and features of dataset
print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

# using PCA to help to visualize data in 2D space
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)

plt.scatter(reduced_data[:,0],reduced_data[:,1],c=kmeans.labels_)

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=3,
            color="b", zorder=10)
plt.title("K-means clustering on the digits dataset")
plt.show()