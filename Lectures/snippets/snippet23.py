import numpy as np
from ipywidgets import *
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML, Markdown
import math

w0 = FloatSlider(min=-6, max=6, step=0.1, value=-1.2, description = 'w0', 
                    continuous_update=False, layout=Layout(width='250px'))

w1 = FloatSlider(min=-2, max=2, step=0.1, value=-0.8, description = 'w1',
                    continuous_update=False, layout=Layout(width='250px'))

w2 = FloatSlider(min=-2, max=2, step=0.1, value=0.5, description = 'w2',
                    continuous_update=False, layout=Layout(width='250px'))


x1 = np.array([1, 2, 2, 4, 5])
x2 = np.array([5, 8, 3, 1, 2])
c = np.array(['r','b','b','r','b'])

ymax = 10
xmax = 6


def neuron_model(w0, w1, w2):
    
    #x = np.array([1, 2, 4, 5, 6])
    #y = np.array([4, 3, 5, 1, 2])
    #c = np.array(['b','b','r','b','r'])

    plt.close()
    plt.rcParams["figure.figsize"] = [6,5]
    plt.axis((0,xmax,0,ymax))
    plt.xlabel('x1', fontsize=20)
    plt.ylabel('x2', fontsize=20)
    
    plt.fill([0,xmax,xmax,0],[0,0,-(w1*xmax + w0)/w2,-w0/w2],'r',alpha=0.2, zorder=1)
    plt.fill([0,xmax,xmax,0],[ymax,ymax,-(w1*xmax + w0)/w2,-w0/w2],'b',alpha=0.2, zorder=1)
    
    plt.plot([0,6],[-w0/w2,-(w1*6 + w0)/w2 ])
    plt.scatter(x1, x2, c=c, zorder=2, s=120, edgecolors='k')
    plt.show()
    
def table(w0, w1, w2):
    z = w0 + w1*x1 + w2*x2
    p = 1 / (1 + np.exp(-z))
    pi = np.where(c == 'b', p, 1 - p)
    lik = np.product(pi)
    
    df = pd.DataFrame({'x1':x1, 'x2':x2, 'col':c, 
                       'p':np.round(p,3), 'pi':np.round(pi,3)}, 
                      columns=['x1','x2','col','p','pi'])
    
    styles = [
        dict(selector="th", props=[("font-size", "150%"), ("text-align", "center")]),
        dict(selector="td", props=[("font-size", "150%"), ("text-align", "center")])
    ]
    
    display(df.style.set_table_styles(styles))

    
def scores(w0, w1, w2):
    z = w0 + w1*x1 + w2*x2
    p = 1 / (1 + np.exp(-z))
    pi = np.where(c == 'b', p, 1 - p)
    lik = np.product(pi)
    
    print("")
    display(Markdown("**<span style=\"font-size:1.6em;\">Likelihood = " 
                     + str(round(lik*100,2)) + "%</span>**"))
    display(Markdown("**<span style=\"font-size:1.6em;\">Loss = " 
                     + str(round(-math.log(lik),4)) + "</span>**"))
    
    


cdict = {'w0':w0, 'w1':w1, 'w2':w2}

plot_out1 = interactive_output(neuron_model, cdict)
calc_out = interactive_output(table, cdict)
score_out = interactive_output(scores, cdict)

ui = VBox([w0, w1, w2, score_out])

display(HBox([ui, plot_out1, calc_out]))


