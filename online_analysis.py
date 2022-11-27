import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
#import plotpy   as py
#import plotpy.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import scikitplot as skplt

data = pd.read_csv('online_shoppers_intention.csv')

missing= data.isnull().sum()
data.fillna(0, inplace=True)

x = data.iloc[:,[5,6]].values
print(x.shape)
# Applyinng K-elbow method
wc =[]
for i in range(1,11):
    km = KMeans(n_clusters =i , init= 'k-means++', max_iter=200, n_init=10, random_state=0, algorithm='full', tol =0.001)
    km.fit(x)
    labels = km.labels_
    wc.append(km.inertia_)
    # inertia_float
    # Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.

"""Visualize the result
plt.rcParams['figure.figsize']=(13,7)
plt.plot(range(1,11),wc)
plt.grid()
plt.tight_layout()
plt.title('Elbow method', fontsize=20)
plt.xlabel('Number of clusters')
plt.ylabel('ws') 
plt.savefig("K-elbow method")
"""

# the maximum curvature is at the second index, that is, the number of optimal clustering groups for the duration of the product and the bounce rates is 2
km = KMeans(n_clusters=2, init='k-means++', max_iter=500, n_init=10, random_state=0)
y_means=km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 50,norm=[0,0.5], c = 'yellow', label = 'Uninterested Customers')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 50,norm=[0,0.5], c = 'pink', label = 'Target Customers')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')
plt.title('Product-related vs bounce rate', fontsize=20)
plt.grid()
plt.xlabel('product related duration')
plt.ylabel('bounce rate')
plt.legend()
plt.savefig('Product-related vs bounce rate')


### Evaluating
le = LabelEncoder()
labels_true = le.fit_transform(data['Revenue'])

# get predicted clustering result label
labels_pred = y_means

# print adjusted rand index, which measures the similarity of the two assignments
from sklearn import metrics
score = metrics.adjusted_rand_score(labels_true, labels_pred)
print("Adjusted rand index: ")
print(score)

# print confusion matrix
#cm = metrics.plot_confusion_matrix(None, labels_true, labels_pred)
#print(cm)

plt_1 = skplt.metrics.plot_confusion_matrix(labels_true, labels_pred, normalize=False)
plt.savefig('Confusion_matrix')
plt_2 = skplt.metrics.plot_confusion_matrix(labels_true, labels_pred, normalize=True)
plt.savefig('Confusion_matrix_inn_percent')
