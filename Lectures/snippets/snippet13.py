from matplotlib.colors import ListedColormap

np.random.seed(1)

L1 = np.array([2,2])
L2 = np.array([4,3])
L = np.vstack([L1,L2])
X1a, y1a = make_circles(n_samples=20, noise=0.1, factor=0.4)
X1b, y1b = make_circles(n_samples=20, noise=0.1, factor=0.4)
X1c, y1c = make_blobs(n_samples=10, centers=1, cluster_std=0.3)
X1d, y1d = make_blobs(n_samples=10, centers=1, cluster_std=0.3)
X1a = X1a + L1
X1b = X1b + L2
X1c = X1c + np.array([6.5,8.5])
X1d = X1d + np.array([7.5,7.5])
X1 = np.vstack([X1a, X1b, X1c, X1d])
y1 = np.hstack([y1a, y1b, y1c, y1d])


g = FloatSlider(min=0.5, max=10, step=0.5, value=5, description = 'g', 
                continuous_update=False, layout=Layout(width='275px'))

def svm_plot_1(g):

    plt.figure(figsize = [12,5])
    
    plt.subplot(1, 2, 1)
    plt.scatter(X1[:, 0], X1[:, 1],  c=y1, s=80, edgecolor='k', cmap='rainbow')
    plt.scatter(L1[0], L1[1], c='darkorange', s=120, edgecolor='k', marker='D')
    plt.scatter(L2[0], L2[1], c='darkgreen', s=120, edgecolor='k', marker='D')


    xticks = np.linspace(0, 10, 100)
    yticks = np.linspace(0, 10, 100)
    grid_pts = np.transpose([np.tile(xticks,100), np.repeat(yticks,100)])
    D1 = np.exp(-g*(np.sum((grid_pts - L1)**2, axis=1)**0.5).reshape(100,100))
    D2 = np.exp(-g*(np.sum((grid_pts - L2)**2, axis=1)**0.5).reshape(100,100))
    
    base = plt.get_cmap('Oranges')
    subset = base(np.linspace(0.45, 0.75))
    cm0 = ListedColormap(subset, name='Custom')
    
    base = plt.get_cmap('Greens')
    subset = base(np.linspace(0.45, 0.75))
    cm1 = ListedColormap(subset, name='Custom')
    
    
    plt.contour(xticks, yticks, D1, cmap=cm0, levels=[0.001, 0.01, 0.1], linestyles = ['--', '--', '--'], zorder = 4)
    plt.contour(xticks, yticks, D2, cmap=cm1, levels=[0.001, 0.01, 0.1], linestyles = ['--', '--', '--'], zorder = 4)
    plt.xlim(0,6)
    plt.ylim(0,4.5)
    plt.title('Original Features')

    T1 = np.exp(-g*np.sum((X1 - L1)**2, axis=1)**0.5).reshape(-1,1)
    T2 = np.exp(-g*np.sum((X1 - L2)**2, axis=1)**0.5).reshape(-1,1)
    T = np.hstack([T1, T2])
    
    plt.subplot(1, 2, 2)
    plt.scatter(T[:, 0], T[:, 1],  c=y1, s=80, edgecolor='k', cmap='rainbow')
    plt.title('Transformed Features')
    plt.show()

cdict = {'g':g}
plot_out = interactive_output(svm_plot_1, cdict)

display(VBox([g, plot_out]))
