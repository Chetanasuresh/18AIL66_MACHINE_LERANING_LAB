from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
iris=datasets.load_iris()
x=pd.DataFrame(iris.data)
x.columns=['Sepal_length','Sepal_width','Petal_length','Petal_width']
y=pd.DataFrame(iris.target)
y.columns=['Targets']
model=KMeans(n_clusters=3)
model.fit(x)
plt.figure(figsize=(14,7))
colormap=np.array(['red','lime','black'])
plt.subplot(1,3,1)
plt.scatter(x.Petal_length,x.Petal_width,c=colormap[y.Targets],s=40)
plt.title('Real Cluster')
plt.xlabel('Petal length')
plt.subplot(1,3,2)
plt.scatter(x.Petal_length,x.Petal_width,c=colormap[model.labels_],s=40)
plt.title('KMeans Clustering')
plt.xlabel('Petal length')
plt.ylabel('Petal Width')
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=3, random_state=0).fit(x)
y_cluster_gmm=gmm.predict(x)
plt.subplot(1,3,3)
plt.title('GMM Clustering')
plt.xlabel('Petal length')
plt.ylabel('Petal Width')
plt.scatter(x.Petal_length,x.Petal_width,c=colormap[y_cluster_gmm])
print('Observation:The GMM using EM Algorithm based clustering matched the true labels more closely than kmeans.')
