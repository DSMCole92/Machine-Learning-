plt.figure(figsize=[12,6])

plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1])

plt.plot([mu[0] + Z[11,0]*dx1, mu[0] + Z[11,0]*dx1 + Z[11,1]*dx2],
         [mu[1] + Z[11,0]*dy1, mu[1] + Z[11,0]*dy1 + Z[11,1]*dy2], linewidth=7, c='k')
plt.plot([mu[0] + Z[11,0]*dx1, mu[0] + Z[11,0]*dx1 + Z[11,1]*dx2],
         [mu[1] + Z[11,0]*dy1, mu[1] + Z[11,0]*dy1 + Z[11,1]*dy2], linewidth=4, c='turquoise')

plt.plot([mu[0] + Z[2,0]*dx1, mu[0] + Z[2,0]*dx1 + Z[2,1]*dx2],
         [mu[1] + Z[2,0]*dy1, mu[1] + Z[2,0]*dy1 + Z[2,1]*dy2], linewidth=7, c='k')
plt.plot([mu[0] + Z[2,0]*dx1, mu[0] + Z[2,0]*dx1 + Z[2,1]*dx2],
         [mu[1] + Z[2,0]*dy1, mu[1] + Z[2,0]*dy1 + Z[2,1]*dy2], linewidth=4, c='turquoise')

plt.plot([mu[0], mu[0] + Z[11,0]*dx1], [mu[1], mu[1] + Z[11,0]*dy1], linewidth=7, c='k')
plt.plot([mu[0], mu[0] + Z[11,0]*dx1], [mu[1], mu[1] + Z[11,0]*dy1], linewidth=4, c='lightcoral')

plt.scatter(mu[0], mu[1], c='darkorange', edgecolor='k', s=160, zorder=4)
plt.scatter(X[2,0], X[2,1], c='violet', edgecolor='k', s=160, zorder=3)
plt.scatter(X[11,0], X[11,1], c='gold', edgecolor='k',s=160, zorder=3)

plt.xlim([-4,4])
plt.ylim([-4,4])
plt.title('Original Data')
plt.xlabel('X0')
plt.ylabel('X1')

plt.subplot(1,2,2)
plt.scatter(Z[:,0], Z[:,1])
plt.scatter([0], [0], c='darkorange', edgecolor='k', s=160)
plt.scatter(Z[2,0], Z[2,1], c='violet', edgecolor='k', s=160)
plt.scatter(Z[11,0], Z[11,1], c='gold', edgecolor='k', s=160)
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.title('Transformed Data')
plt.xlabel('Z0')
plt.ylabel('Z1')
plt.show()