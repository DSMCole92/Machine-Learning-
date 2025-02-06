import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

#sd = np.random.choice(range(0,200))
sd = 95
np.random.seed(sd)
#print("Seed:", sd)


X, y_true = make_blobs(n_samples=200, centers=4,
                       cluster_std=1.60, random_state=sd)



n1 = 100
x1 = np.random.normal(3, 0.5, n1).reshape(n1,1)
y1 = np.random.normal(6, 0.8, n1).reshape(n1,1)

n2 = 50
x2 = np.random.normal(5.5, 0.5, n2).reshape(n2,1)
y2 = np.random.normal(4.5, 0.5, n2).reshape(n2,1)

n3 = 50
x3 = np.random.normal(6.75, 0.5, n3).reshape(n3,1)
y3 = np.random.normal(5.75, 0.5, n3).reshape(n3,1)

n4 = 50
x4 = np.random.normal(6, 0.5, n4).reshape(n4,1)
y4 = np.random.normal(3, 0.5, n4).reshape(n4,1)

n5 = 50
x5 = np.random.normal(8.5, 0.5, n5).reshape(n5,1)
y5 = np.random.normal(3, 0.5, n5).reshape(n5,1)

X1 = np.vstack((x1, x2, x3, x4, x5))
X2 = np.vstack((y1, y2, y3, y4, y5))

X = np.hstack((X1, X2))


kmeans3 = KMeans(n_clusters=3)
kmeans3.fit(X)
y_kmeans3 = kmeans3.predict(X)

kmeans4 = KMeans(n_clusters=4)
kmeans4.fit(X)
y_kmeans4 = kmeans4.predict(X)

kmeans5 = KMeans(n_clusters=5)
kmeans5.fit(X)
y_kmeans5 = kmeans5.predict(X)


plt.close()
plt.figure(figsize=(9,9))

plt.subplot(2,2,1)
plt.scatter(X[:,0], X[:,1], s=20, cmap='rainbow', edgecolor='k')
plt.title('Original Data')

plt.subplot(2,2,2)
plt.scatter(X[:,0], X[:,1],c=y_kmeans3, s=20, cmap='rainbow', edgecolor='k')
plt.title('K-Means (K = 3)')

plt.subplot(2,2,3)
plt.scatter(X[:,0], X[:,1],c=y_kmeans4, s=20, cmap='rainbow', edgecolor='k')
plt.title('K-Means (K = 4)')

plt.subplot(2,2,4)
plt.scatter(X[:,0], X[:,1],c=y_kmeans5, s=20, cmap='rainbow', edgecolor='k')
plt.title('K-Means (K = 5)')

plt.show()
