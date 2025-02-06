np.random.seed(3158)
X2_temp, y2 = make_circles(n_samples=100, noise=0.2, factor=0.05)
X2 = np.array([2 * X2_temp[:,0] + 6 * X2_temp[:,1], 
               2 * X2_temp[:,0] - 3 * X2_temp[:,1]]).T

D = IntSlider(min=1, max=4, step=1, value=1, description = 'D', 
                    continuous_update=False, layout=Layout(width='275px'))
C = FloatSlider(min=0.01, max=20, step=0.01, value=20, description = 'C', 
                    continuous_update=False, layout=Layout(width='275px'))


s1 = Checkbox(value=False, description='Show Margins', disable=False)
s2 = Checkbox(value=False, description='Show Support', disable=False)

def svm_plot_2(D, C, s1, s2):

    mod_02 = SVC(C=C, kernel='poly', degree=D, gamma='auto')
    mod_02.fit(X2, y2)
    
    
    plot_regions(mod_02, X2, y2, 200, display=False, fig_size=[8,6])
    
    if(s2):
        plt.scatter(mod_02.support_vectors_[:, 0], mod_02.support_vectors_[:, 1], s=300,  
                    linewidth=1, edgecolors='k', zorder=4, facecolors='none')
    
    if(s1):
        xticks = np.linspace(np.min(X2[:,0])-0.5, np.max(X2[:,0])+0.5, 100)
        yticks = np.linspace(np.min(X2[:,1])-0.5, np.max(X2[:,1])+0.5, 100)
        grid_pts = np.transpose([np.tile(xticks,100), np.repeat(yticks,100)])
        P = mod_02.decision_function(grid_pts).reshape(100,100)
        plt.contour(xticks, yticks, P, colors='k', levels=[-1,0,1], 
                    linestyles = ['--', '-', '--'], zorder = 4)
        
    plt.show()


cdict = {'D':D, 'C':C, 's1':s1, 's2':s2}
plot_out = interactive_output(svm_plot_2, cdict)
ui = VBox([D, C, s1, s2])
display(HBox([ui, plot_out]))