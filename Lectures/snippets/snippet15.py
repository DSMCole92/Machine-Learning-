np.random.seed(159)
X2, y2 = make_moons(n_samples=100, noise=0.1)

D = IntSlider(min=1, max=10, step=1, value=1, description = 'D', 
                    continuous_update=False, layout=Layout(width='275px'))
C = FloatSlider(min=0.01, max=100, step=0.01, value=20, description = 'C', 
                    continuous_update=False, layout=Layout(width='275px'))
G = FloatSlider(min=0.1, max=10, step=0.1, value=0.5, description = 'G', 
                    continuous_update=False, layout=Layout(width='275px'))

s1 = Checkbox(value=False, description='Show Margins', disable=False)
s2 = Checkbox(value=False, description='Show Support', disable=False)
s3 = Checkbox(value=False, description='RBF Kernel', disable=False)

def svm_plot(D, C, G, s1, s2, s3):

    mapping = 'poly'
    if(s3):
        mapping = 'rbf'
        
    mod_02 = SVC(C=C, kernel=mapping, degree=D, gamma=G)
    mod_02.fit(X2,y2)
    plot_regions(mod_02, X2, y2, 200, display=False)
    
    if(s2):
        plt.scatter(mod_02.support_vectors_[:, 0], mod_02.support_vectors_[:, 1], s=300,  
                    linewidth=1, edgecolors='k', zorder=4, facecolors='none')
    
    if(s1):
        xticks = np.linspace(np.min(X2[:,0]), np.max(X2[:,0]), 100)
        yticks = np.linspace(np.min(X2[:,1]), np.max(X2[:,1]), 100)
        grid_pts = np.transpose([np.tile(xticks,100), np.repeat(yticks,100)])
        P = mod_02.decision_function(grid_pts).reshape(100,100)
        plt.contour(xticks, yticks, P, colors='k', levels=[-1,0,1], 
                    linestyles = ['--', '-', '--'], zorder = 4)
        
    plt.show()


cdict = {'D':D, 'C':C, 'G':G, 's1':s1, 's2':s2, 's3':s3}
plot_out = interactive_output(svm_plot, cdict)
ui = VBox([D, C, G, s1, s2, s3])
display(HBox([ui, plot_out]))