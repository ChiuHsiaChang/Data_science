
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN,Birch
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples,silhouette_score,calinski_harabasz_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


# In[2]:


db = pd.read_csv(r'D:\2020\資料科學\hw2\data.csv',engine='python')
db.info()


# In[3]:


db.head()


# In[4]:


db=db.dropna()
db=db.drop(["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9",
            "feature17","feature18","feature19","id"],axis=1)
# db=db.drop(["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","id"],axis=1)
# db=db.drop(["feature3","feature4","feature5","id"],axis=1)
#db=preprocessing.scale(db)


# In[5]:


db.head()


# In[6]:


minmax = preprocessing.MinMaxScaler()
db = minmax.fit_transform(db)


# In[8]:


#------------------------------------------------------K-means
for n_clusters in range(3,15,1):
    n_clusters=n_clusters
#     fig,(ax1,ax2) = plt.subplots(1,2)
    fig = plt.figure()
    ax1 = plt.subplot(2,1,1)
    fig.set_size_inches(12,7)
    ax1.set_xlim([-0.1,1])
    ax1.set_ylim([0,db.shape[0] + (n_clusters + 1)*10])
    clusterer = KMeans(n_clusters=n_clusters,random_state=10).fit(db)
    cluster_labels=clusterer.labels_
    silhouette_avg = silhouette_score(db,cluster_labels)
    cal=calinski_harabasz_score(db,cluster_labels)
    print(n_clusters,':',cal)
    print("n_cluster = ",n_clusters,'. The average silhouette_score is:', silhouette_avg)
    sample_silhoutte_values = silhouette_samples(db,cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhoutte_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        ax1.axvline(x=silhouette_avg,color='red', linestyle='--')
        color = cm.nipy_spectral(float(i)/n_clusters)
        ax1.fill_betweenx(np.arange(y_lower,y_upper),
                         ith_cluster_silhouette_values,
                         facecolor = color,
                         alpha=0.7
                         )
        ax1.text(-0.05, y_lower+0.5 *size_cluster_i, str(i))
        y_lower=y_upper+10


# In[10]:


#------------------------------------------------------K-means
for n_clusters in range(3,15,1):
    n_clusters=n_clusters
    clusterer = KMeans(n_clusters=n_clusters,random_state=10).fit(db)
    cluster_labels=clusterer.labels_
    silhouette_avg = silhouette_score(db,cluster_labels)
    cal=calinski_harabasz_score(db,cluster_labels)
    print(n_clusters,':',cal)
    print("n_cluster = ",n_clusters,'. The average silhouette_score is:', silhouette_avg)


# In[12]:


#------------------------------------------------------DBSCAN
for e in [0.019,0.021,0.022,0.023,0.024]:
    for m in [0.5,1,2,3]:
        clusterer = DBSCAN(eps =e,min_samples=m).fit(db)
        cluster_labels=clusterer.labels_
        silhouette_avg = silhouette_score(db,cluster_labels)
        cal=calinski_harabasz_score(db,cluster_labels)
        print(e,'_',m,':',cal)
        print("eps_min",e,'_',m,'. The average silhouette_score is:', silhouette_avg)


# In[16]:


#------------------------------------------------------Birch
for n_clusters in [3,5,10,15]:
    for threshold in [0.005,0.1,0.5]:
        for branching_factor in [10,20,30]:
            clusterer = Birch(n_clusters = n_clusters, threshold = threshold, branching_factor = branching_factor).fit(db)
            cluster_labels=clusterer.labels_
            silhouette_avg = silhouette_score(db,cluster_labels)
            cal=calinski_harabasz_score(db,cluster_labels)
            print(n_clusters,'_',threshold,'_',branching_factor,':',cal)
            print(n_clusters,'_',threshold,'_',branching_factor,'. The average silhouette_score is:', silhouette_avg)


# In[27]:


#-----------------------------------------------------------------------Birch
import warnings
warnings.filterwarnings("ignore")
test = pd.read_csv(r'D:\2020\資料科學\hw2\test.csv',engine='python')
test=test.drop(['index'],axis=1)
# for e in np.arange([0.019,0.021,0.022,0.023,0.024]):
#for n_clusters in np.arange(20,15,1):
for n_clusters in [10,11,12,13,14,15,16,17,18,19]:
    for threshold in [0.005]:
        for branching_factor in [10]:
            cluster = Birch(n_clusters = n_clusters, threshold = threshold, branching_factor = branching_factor).fit(db)
            out=cluster.fit_predict(db)
            a=list(test.iloc[:, 0])
            b=list(test.iloc[:, 1])
            y_test=[]
            for j in range(400):
                if(out[a[j]]==out[b[j]]):
                    ans=1
                else:
                    ans=0
                y_test.append(ans)
            dataframe = pd.DataFrame({'ans':y_test})
            n_clusters_s=str(n_clusters)
            threshold_s=str(threshold)
            branching_factor_s=str(branching_factor)
            path=r'D:\2020\資料科學\hw2\Birch' 
            path += '\\'+n_clusters_s+'_'+threshold_s+'_'+branching_factor_s+ '.csv'
            cal=calinski_harabasz_score(db,out)
            print(n_clusters,'_',threshold,'_',branching_factor,':',cal)
            dataframe.to_csv(path,index=True,index_label='index',sep=',')

