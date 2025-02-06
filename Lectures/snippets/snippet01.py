# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd

###################################
# Generate Data 
###################################

sd = np.random.choice(range(0,2000))
sd = 1956
np.random.seed(sd)
#print("Seed:", sd)

N = 10
x = np.linspace(2,8,N)
y =  16 - (x - 6.5)**2 + np.random.normal(0, 1, N)

xgrid = np.linspace(0,10,50)

###################################
# Linear Model
###################################
from sklearn.linear_model import LinearRegression

m1 = LinearRegression()
m1.fit(x.reshape(N,1),y)
pred1 = m1.predict(xgrid.reshape(50,1))

###################################
# Quadratic Model
###################################
from sklearn.preprocessing import PolynomialFeatures

pt = PolynomialFeatures(2)
Xpoly = pt.fit_transform(x.reshape(N,1))
m2 = LinearRegression()
m2.fit(Xpoly,y)
pred2 = m2.predict(pt.transform(xgrid.reshape(50,1)))


###################################
# Degree 10 Model
###################################
from sklearn.preprocessing import PolynomialFeatures

pt = PolynomialFeatures(10)
Xpoly = pt.fit_transform(x.reshape(N,1))
m3 = LinearRegression()
m3.fit(Xpoly,y)
pred3 = m3.predict(pt.transform(xgrid.reshape(50,1)))

###################################
# Piecewise Linear
###################################
from scipy.optimize import minimize

b = [6,-9,4.5,-3]

def pred(b, x0):
    p1 = b[1] + b[2]*x0
    p2 = b[1] + b[0]*(b[2] - b[3]) + b[3]*x0
    p = np.where(x0 < b[0], p1, p2)
    return p

def sse(b):    
    p = pred(b, x)
    e = p - y
    return np.sum(e**2)

min_results = minimize(sse, b)
b_opt = min_results.x
pred4 = pred(b_opt, xgrid)
           
###################################
# KNN Model
###################################
from sklearn.neighbors import KNeighborsRegressor

m5 = KNeighborsRegressor(3)
m5.fit(x.reshape(N,1),y)
pred5 = m5.predict(xgrid.reshape(50,1))

###################################
# Plot
###################################

x0 = 0
x1 = 10
y0 = 0
y1 = 20

plt.close()
plt.rcParams["figure.figsize"] = [12,8]

plt.subplot(2,3,1)
plt.scatter(x, y)
plt.xlim(x0,x1)
plt.ylim(y0,y1)
plt.title('Original Data')

plt.subplot(2,3,2)
plt.plot(xgrid, pred1, c='darkorange', zorder=1)
plt.scatter(x, y, zorder=2)
plt.xlim(x0,x1)
plt.ylim(y0,y1)
plt.title('Model 1: Linear Regression')

plt.subplot(2,3,3)
plt.plot(xgrid, pred2, c='darkorange', zorder=1)
plt.scatter(x, y, zorder=2)
plt.xlim(x0,x1)
plt.ylim(y0,y1)
plt.title('Model 2: Quadratic Model')

plt.subplot(2,3,4)
plt.plot(xgrid, pred3, c='darkorange', zorder=1)
plt.scatter(x, y, zorder=2)
plt.xlim(x0,x1)
plt.ylim(y0,y1)
plt.title('Model 3: Degree 10 Poly')

plt.subplot(2,3,5)
plt.plot(xgrid, pred4, c='darkorange', zorder=1)
plt.scatter(x, y, zorder=2)
plt.xlim(x0,x1)
plt.ylim(y0,y1)
plt.title('Model 4: PW-Linear Regression')

plt.subplot(2,3,6)
plt.plot(xgrid, pred5, c='darkorange', zorder=1)
plt.scatter(x, y, zorder=2)
plt.xlim(x0,x1)
plt.ylim(y0,y1)
plt.title('Model 5: 3-Nearest Neighbors')

plt.show()








#
#
#
#
####################################
## Generate Data 
####################################
#
##sd = np.random.choice(range(0,200))
#sd = 194
#np.random.seed(sd)
##print("Seed:", sd)
#
#N = 80
#x1 = np.random.uniform(0.5,9.5,N)
#x2 = np.random.uniform(0.5,9.5,N)
#
#clrs = np.array(['b','r'])
#
#z = 0.05 * (x1**2 + 4*(x2-3)**2 - 30)
#prob = 1 / (1 + np.exp(-z))
#rolls = np.random.uniform(0,1,N)
#y = (prob > rolls).astype(int)
#
####################################
## Generate Scatterplot
####################################
#
#X = pd.DataFrame({'x1':x1, 'x2':x2})
#
#df = pd.DataFrame({'x1':x1, 'x2':x2, 'y':y})
#df.to_csv('data.txt', sep='\t')
#
#plt.close()
#plt.rcParams["figure.figsize"] = [5,5]
#plt.scatter(x1,x2,c=clrs[y])
#plt.show()
#
####################################
## Create grid for heatmap
####################################
#
#xTicks = np.linspace(0, 10, 1000)
#yTicks = np.linspace(0, 10, 1000)
#xGrid, yGrid = np.meshgrid(xTicks, yTicks)
#grid = np.vstack((xGrid.flatten(), yGrid.flatten())).T
#
####################################
## Prepare plots
####################################
#
#plt.close()
#plt.rcParams["figure.figsize"] = [15,10]
#plt.figure()
#
#
####################################
## Logistic Regression
####################################
#
#mod01 = linear_model.LogisticRegression(C = 1000000)
#mod01.fit(X,y)
#
#prob = mod01.predict_proba(X=grid)
#sel = (prob[:,1] > 0.5).astype(int)
#selGrid = sel.reshape(xGrid.shape)
#
#myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
#plt.subplot(2,3,1)
#plt.pcolormesh(xTicks, yTicks, selGrid, cmap=myCmap)
#plt.scatter(x1,x2,c=clrs[y])
#plt.title('Logistic Regression')
#
####################################
## SVM, RBF Kernel
####################################
#
#mod02 = svm.SVC(kernel='rbf', gamma=0.9, C=1.0)
#mod02.fit(X,y)
#
#pred = mod02.predict(X=grid)
#predGrid = pred.reshape(xGrid.shape)
#
#myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
#plt.subplot(2,3,2)
#plt.pcolormesh(xTicks, yTicks, predGrid, cmap=myCmap)
#plt.scatter(x1,x2,c=clrs[y])
#plt.title('SVM (RBF Kernel')
#
####################################
## SVM, Polynomial Kernel
####################################
#
#mod03 = svm.SVC(kernel='poly', degree=3, C=1.0)
#mod03.fit(X,y)
#
#pred = mod03.predict(X=grid)
#predGrid = pred.reshape(xGrid.shape)
#
#myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
#plt.subplot(2,3,3)
#plt.pcolormesh(xTicks, yTicks, predGrid, cmap=myCmap)
#plt.scatter(x1,x2,c=clrs[y])
#plt.title('SVM (Polynomial Kernel)')
#
####################################
## KNN, N=1
####################################
#
#mod04 = neighbors.KNeighborsClassifier(n_neighbors=1)
#mod04.fit(X,y)
#
#pred = mod04.predict(X=grid)
#predGrid = pred.reshape(xGrid.shape)
#
#myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
#plt.subplot(2,3,4)
#plt.pcolormesh(xTicks, yTicks, predGrid, cmap=myCmap)
#plt.scatter(x1,x2,c=clrs[y])
#plt.title('KNN (K=1)')
#
####################################
## KNN, N=3
####################################
#
#mod04 = neighbors.KNeighborsClassifier(n_neighbors=3)
#mod04.fit(X,y)
#
#pred = mod04.predict(X=grid)
#predGrid = pred.reshape(xGrid.shape)
#
#myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
#plt.subplot(2,3,5)
#plt.pcolormesh(xTicks, yTicks, predGrid, cmap=myCmap)
#plt.scatter(x1,x2,c=clrs[y])
#plt.title('KNN (K=3)')
#
#
#
####################################
## Decision Tree
####################################
#
#mod05 = tree.DecisionTreeClassifier()
#mod05.fit(X,y)
#
#pred = mod05.predict(X=grid)
#predGrid = pred.reshape(xGrid.shape)
#
#myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
#plt.subplot(2,3,6)
#plt.pcolormesh(xTicks, yTicks, predGrid, cmap=myCmap)
#plt.scatter(x1,x2,c=clrs[y])
#plt.title('Decision Tree')
#
#plt.show()
#
#
#
