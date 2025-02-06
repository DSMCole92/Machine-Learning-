np.random.seed(159)
X3, y3 = make_moons(n_samples=100, noise=0.1)

D = IntSlider(min=1, max=10, step=1, value=1, description = 'D', 
                    continuous_update=False, layout=Layout(width='275px'))
C = FloatSlider(min=0.01, max=100, step=0.01, value=20, description = 'C', 
                    continuous_update=False, layout=Layout(width='275px'))

s1 = Checkbox(value=False, description='Show Margins', disable=False)
s2 = Checkbox(value=False, description='Show Support', disable=False)

def svm_plot_3(D, C, s1, s2):

    mod_03 = SVC(C=C, kernel='poly', degree=D, gamma='auto')
    mod_03.fit(X3, y3)
    plot_regions(mod_03, X3, y3, 200, display=False, fig_size=[8,6])
    
    if(s2):
        plt.scatter(mod_03.support_vectors_[:, 0], mod_03.support_vectors_[:, 1], s=300,  
                    linewidth=1, edgecolors='k', zorder=4, facecolors='none')
    
    if(s1):
        xticks = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
        yticks = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
        grid_pts = np.transpose([np.tile(xticks,100), np.repeat(yticks,100)])
        P = mod_03.decision_function(grid_pts).reshape(100,100)
        plt.contour(xticks, yticks, P, colors='k', levels=[-1,0,1], 
                    linestyles = ['--', '-', '--'], zorder = 4)
        
    plt.show()


cdict = {'D':D, 'C':C, 's1':s1, 's2':s2}
plot_out = interactive_output(svm_plot_3, cdict)
ui = VBox([D, C, s1, s2])
display(HBox([ui, plot_out]))