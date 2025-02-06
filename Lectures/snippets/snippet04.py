import numpy as np
from ipywidgets import *
from IPython.display import display, HTML, Markdown
import pandas as pd
import matplotlib.pyplot as plt

#sd = np.random.choice(range(0,200))
sd = 164
np.random.seed(sd)
#print("Seed:", sd)
N_s4 = 12
x_s4 = np.random.uniform(low=0,high=10,size=N_s4)
x_s4.sort()

y_s4 = 5 + 1.4 * x_s4 + np.random.normal(0,2.5,N_s4)

def regression_example(b, m, e):
    yhat = b + m * x_s4
    
    plt.rcParams["figure.figsize"] = [5.5,5.5]
    plt.plot([0,10],[b,10*m+b],c='purple')
    
    if e:
        for i in range(len(x_s4)):
            plt.plot([x_s4[i],x_s4[i]],[y_s4[i],yhat[i]],c='black',
                     lw=0.75,zorder=1)
    
    plt.scatter(x_s4,y_s4,zorder=2)        

    plt.axis((0,10,0,25))
    plt.show()


def sse(b, m, e):
    yhat = b + m * x_s4 
    errors = y_s4 - yhat
    sq_errors = errors**2

    SSE = np.sum(sq_errors)

    print('')
    
    sp1 = "**<span style=\"font-size:1.4em;\">"
    sp2 = "</span>**"
    
    display(Markdown(sp1 + "Loss Function:" + sp2))
    display(Markdown(sp1 + "Sum of Squared " + sp2))
    display(Markdown(sp1 + "Errors = " + str(round(SSE,2)) + sp2))
                     
    
    

def table(b, m, e):   
    yhat = b + m * x_s4 
    errors = y_s4 - yhat
    sq_errors = errors**2
    
    df = pd.DataFrame({'x':np.round(x_s4,3), 'y':np.round(y_s4,3), 
                       'yhat':np.round(yhat,3), 'error':np.round(errors,3), 
                       'sq_error':np.round(sq_errors,3)})
    
    styles = [
        dict(selector="th", props=[("font-size", "100%"), ("text-align", "center")]),
        dict(selector="td", props=[("font-size", "100%"), ("text-align", "center")])
    ]
   
    display(df.style.set_table_styles(styles))

    
#_ = widgets.interact(regressionExample,
#                     b=widgets.FloatSlider(min=-2,max=10,step=0.1,value=2,continuous_update=False),
#                     m=widgets.FloatSlider(min=-2,max=2,step=0.01,value=0,continuous_update=False),
#                     e=widgets.Checkbox(value=False,description='Show Errors',disable=False))

b = FloatSlider(min=-2, max=10, step=0.1, value=2, 
                continuous_update=False, layout=Layout(width='200px'))
m = FloatSlider(min=-2, max=2, step=0.01, value=0, 
                continuous_update=False, layout=Layout(width='200px'))
e = Checkbox(value=False, description='Show Errors', disable=False)

cdict = {'b':b, 'm':m, 'e':e}

plot_out = interactive_output(regression_example, cdict)
sse_out = interactive_output(sse, cdict)
table_out = interactive_output(table, cdict)

ui = VBox([b, m, e, sse_out])

display(HBox([ui, plot_out, table_out]))
