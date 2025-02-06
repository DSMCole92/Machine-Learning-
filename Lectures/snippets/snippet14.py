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

D = IntSlider(min=1, max=3, step=1, value=1, description = 'D', 
                    continuous_update=False, layout=Layout(width='275px'))
C = FloatSlider(min=0.01, max=100, step=0.01, value=20, description = 'C', 
                    continuous_update=False, layout=Layout(width='275px'))
G = FloatSlider(min=0.1, max=20, step=0.1, value=0.5, description = 'G', 
                    continuous_update=False, layout=Layout(width='275px'))

s1 = Checkbox(value=False, description='Show Margins', disable=False)
s2 = Checkbox(value=False, description='Show Support', disable=False)
s3 = Checkbox(value=False, description='RBF Kernel', disable=False)

def svm_plot(D, C, G, s1, s2, s3):

    mapping = 'poly'
    if(s3):
        mapping = 'rbf'
        
    mod_01 = SVC(C=C, kernel=mapping, degree=D, gamma=G)
    mod_01.fit(X1, y1)
    plot_regions(mod_01, X1, y1, 200, display=False)
    
    if(s2):
        plt.scatter(momod_01d.support_vectors_[:, 0], mod_01.support_vectors_[:, 1], s=300,  
                    linewidth=1, edgecolors='k', zorder=4, facecolors='none')
    
    if(s1):
        xticks = np.linspace(np.min(X1[:,0]), np.max(X1[:,0]), 100)
        yticks = np.linspace(np.min(X1[:,1]), np.max(X1[:,1]), 100)
        grid_pts = np.transpose([np.tile(xticks,100), np.repeat(yticks,100)])
        P = mod_01.decision_function(grid_pts).reshape(100,100)
        plt.contour(xticks, yticks, P, colors='k', levels=[-1,0,1], 
                    linestyles = ['--', '-', '--'], zorder = 4)
        
    plt.show()


cdict = {'D':D, 'C':C, 'G':G, 's1':s1, 's2':s2, 's3':s3}
plot_out = interactive_output(svm_plot, cdict)
ui = VBox([D, C, G, s1, s2, s3])
display(HBox([ui, plot_out]))