np.random.seed(7530)
X3, y3 = make_blobs(n_samples=100, centers=6, n_features=2, cluster_std=2)
y3 = np.where(y3%2 == 0, 0, np.where(y3%2 == 1, 1, y3))

D = IntSlider(min=1, max=5, step=1, value=1, description = 'D', 
                    continuous_update=False, layout=Layout(width='275px'))
C = FloatSlider(min=0.1, max=5, step=0.1, value=0.1, description = 'C', 
                    continuous_update=False, layout=Layout(width='275px'))
G = FloatSlider(min=0.01, max=5, step=0.01, value=0.5, description = 'G', 
                    continuous_update=False, layout=Layout(width='275px'))

s1 = Checkbox(value=False, description='Show Margins', disable=False)
s2 = Checkbox(value=False, description='Show Support', disable=False)
s3 = Checkbox(value=False, description='RBF Kernel', disable=False)

def svm_plot(D, C, G, s1, s2, s3):

    mapping = 'poly'
    if(s3):
        mapping = 'rbf'
        
    mod_03 = SVC(C=C, kernel=mapping, degree=D, gamma=G)
    mod_03.fit(X3, y3)
    plot_regions(mod_03, X3, y3, 100, display=False)
    
    if(s2):
        plt.scatter(mod_03.support_vectors_[:, 0], mod_03.support_vectors_[:, 1], s=300,  
                    linewidth=1, edgecolors='k', zorder=4, facecolors='none')
    
    if(s1):
        xticks = np.linspace(np.min(X3[:,0]), np.max(X3[:,0]), 100)
        yticks = np.linspace(np.min(X3[:,1]), np.max(X3[:,1]), 100)
        grid_pts = np.transpose([np.tile(xticks,100), np.repeat(yticks,100)])
        P = mod_03.decision_function(grid_pts).reshape(100,100)
        plt.contour(xticks, yticks, P, colors='k', levels=[-1,0,1], 
                    linestyles = ['--', '-', '--'], zorder = 4)
        
    plt.show()


cdict = {'D':D, 'C':C, 'G':G, 's1':s1, 's2':s2, 's3':s3}
plot_out = interactive_output(svm_plot, cdict)
ui = VBox([D, C, G, s1, s2, s3])
display(HBox([ui, plot_out]))