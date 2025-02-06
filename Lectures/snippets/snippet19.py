np.random.seed(1)
n = 50

# Generate approximation of transformed coordinates
v0 = np.random.normal(0,1.2,[n,1])
v1 = np.random.normal(0,0.6,[n,1])
v2 = np.random.normal(0,0.2,[n,1])
Z_ = np.hstack([v0,v1,v2])

# Approximation of principal components
pc0 = np.array([4,2,1])/np.sqrt(21)
pc1 = np.array([-1,1,2])/np.sqrt(5)
pc2 = np.array([3,-9,6])/np.sqrt(126)

# Generate 'original data'
X = v0*pc0 + v1*pc1 + v2*pc2 + np.array([3,3,3])

mu = np.mean(X, axis=0)

pca = PCA(n_components=3)
Z = pca.fit_transform(X)

pc = pca.components_

plt.close()
fig = plt.figure(figsize=[8,8])
ax = fig.gca(projection='3d')

ax.scatter(X[:,0], X[:,1], X[:,2], s=20, marker = 'o', c='grey', edgecolor='k', label=0)

ax.quiver(mu[0], mu[1], mu[2], pc[0,0], pc[0,1], pc[0,2], color = 'r', length=1)
ax.quiver(mu[0], mu[1], mu[2], pc[1,0], pc[1,1], pc[1,2], color = 'b', length=1)
ax.quiver(mu[0], mu[1], mu[2], pc[2,0], pc[2,1], pc[2,2], color = 'g', length=1)

ax.set_xlim(0,6)
ax.set_ylim(0,6)
ax.set_zlim(0,6)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
plt.show()