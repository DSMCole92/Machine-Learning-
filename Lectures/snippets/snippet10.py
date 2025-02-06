np.random.seed(1)
X1, y1 = make_blobs(n_samples=100, centers=2, cluster_std=2)
X1[0,:] = [-4, -5]
X1[1,:] = [-6, 7]


C = FloatSlider(min=0.01, max=5, step=0.01, value=5, description = 'C', 
                continuous_update=False, layout=Layout(width='275px'))

def svm_plot_1(C):

    # build model
    mod_01 = SVC(C=C, kernel='linear')
    mod_01.fit(X1, y1)
    
    # plot decision regions
    plot_regions(mod_01, X1, y1, 200, display=False, fig_size=[8,6])
    
    # plot support vectors
    plt.scatter(mod_01.support_vectors_[:, 0], mod_01.support_vectors_[:, 1], s=300, 
                linewidth=1, edgecolors='k', zorder=4, facecolors='none')
    
    xticks = np.linspace(np.min(X1[:,0])-1/2, np.max(X1[:,0])+1/2, 100)
    yticks = np.linspace(np.min(X1[:,1])-1/2, np.max(X1[:,1])+1/2, 100)
    grid_pts = np.transpose([np.tile(xticks,100), np.repeat(yticks,100)])
    
    P = mod_01.decision_function(grid_pts).reshape(100,100)
    
    plt.contour(xticks, yticks, P, colors='k', levels=[-1,0,1], linestyles = ['--', '-', '--'], zorder = 4)
    plt.show()


cdict = {'C':C}
plot_out = interactive_output(svm_plot_1, cdict)

display(VBox([C, plot_out]))
