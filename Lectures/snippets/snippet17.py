dx1, dy1 = pc[0,]
dx2, dy2 = pc[1,]
plt.figure(figsize=[6,6])
plt.scatter(X[:,0], X[:,1], c='grey')
plt.scatter(mu[0], mu[1], c='darkorange', edgecolor='k', s=160, zorder=4)
plt.plot([mu[0], mu[0] + dx1], [mu[1], mu[1] + dy1], linewidth=7, c='k')
plt.plot([mu[0], mu[0] + dx1], [mu[1], mu[1] + dy1], linewidth=4, c='lightcoral')
plt.plot([mu[0], mu[0] + dx2], [mu[1], mu[1] + dy2], linewidth=7, c='k')
plt.plot([mu[0], mu[0] + dx2], [mu[1], mu[1] + dy2], linewidth=4, c='turquoise')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.show()