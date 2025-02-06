import numpy as np
import matplotlib.pyplot as plt
from ClassificationPlotter import plot_regions

###################################
# Generate Data 
###################################


#sd = np.random.choice(range(0,2000))
sd = 623
np.random.seed(sd)
#print("Seed:", sd)

N = 150
x1 = np.random.uniform(0,10, N)
x2 = np.random.uniform(0,10, N)

X = np.hstack((x1.reshape(N,1), x2.reshape(N,1)))

z = 0.2*(x1**2 + x2**2 + 0.5*x1 + 0.5*x2 - x1*x2 - 30)
p = 1 / (1 + np.exp(-z))

#plt.hist(p)
#plt.show()

roll = np.random.uniform(0,1,N)

y = np.where(p < roll, 0, 1)

from matplotlib.colors import ListedColormap
#base = plt.get_cmap('RdYlBu')
#subset = base(np.linspace(0.25, 0.85), 128)
#cm0 = ListedColormap(subset, name='Custom')

top = plt.get_cmap('Oranges_r', 128)
bottom = plt.get_cmap('Blues', 128)
newcolors = np.vstack((top(np.linspace(0.6, 1, 128)),
                       bottom(np.linspace(0, 0.75, 128))))
cm0 = ListedColormap(newcolors, name='OrangeBlue')



###################################
# Logistic Regression
###################################
from sklearn.linear_model import LogisticRegression

m1 = LogisticRegression(solver='lbfgs')
m1.fit(X, y)


###################################
# SVM, rbf
###################################
from sklearn.svm import SVC

m2 = SVC(kernel='poly', degree=3, C=1.0, gamma='auto')
m2.fit(X,y)

###################################
# SVM, rbf
###################################
m3 = SVC(kernel='rbf', gamma=0.3, C=50.0)
m3.fit(X,y)


###################################
# SVM, rbf
###################################
from sklearn.neighbors import KNeighborsClassifier

m4 = KNeighborsClassifier(1)
m4.fit(X,y)


###################################
# Decision Tree
###################################
from sklearn.tree import DecisionTreeClassifier

m5 = DecisionTreeClassifier()
m5.fit(X, y)


#plot_regions(m1, X, y, num_ticks=500, fig_size=(8,6), cmap=cm0, display=False)


###################################
# Plot
###################################
nticks = 200

plt.close()
#plt.rcParams["figure.figsize"] = [12,8]

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.scatter(x1, x2, c=y, edgecolor='k', cmap=cm0)
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Original Data')


plt.subplot(2,3,2)
plot_regions(m1, X, y, num_ticks=nticks, cmap=cm0, close=False, legend=False,
             display=False)
plt.title('Model 1: Logistic Regression')

plt.subplot(2,3,3)
plot_regions(m2, X, y, num_ticks=nticks, cmap=cm0, close=False, legend=False,
             display=False)
plt.title('Model 2: SVM (Poly Kernel)')

plt.subplot(2,3,4)
plot_regions(m3, X, y, num_ticks=nticks, cmap=cm0, close=False, legend=False,
             display=False)
plt.title('Model 3: SVM (RBF Kernel)')

plt.subplot(2,3,5)
plot_regions(m4, X, y, num_ticks=nticks, cmap=cm0, close=False, legend=False,
             display=False)
plt.title('Model 4: 1-Nearest Neighbors')

plt.subplot(2,3,6)
plot_regions(m5, X, y, num_ticks=nticks, cmap=cm0, close=False, legend=False,
             display=False)
plt.title('Model 5: Decision Tree')

plt.show()

import pandas as pd

acc1 = round(m1.score(X, y),3)
acc2 = round(m2.score(X, y),3)
acc3 = round(m3.score(X, y),3)
acc4 = round(m4.score(X, y),3)
acc5 = round(m5.score(X, y),3)

acc = [[acc1, acc2, acc3, acc4, acc5]]
acc = pd.DataFrame(acc)
acc.columns = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']
acc.index = ['Accuracy']

print('\n', acc)