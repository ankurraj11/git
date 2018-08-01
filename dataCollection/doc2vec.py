
# coding: utf-8

# In[72]:


import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import AffinityPropagation
import numpy as np
import pandas as pd
import networkx as nx
import newspaper
from newspaper import Article 


# In[34]:


from os import listdir
from os.path import isfile, join
import itertools
import gensim
from gensim.models import Doc2Vec


# In[35]:


def nearestExempler(exempler, centers):
    dist = []
    for i in range(len(centers)):
        other_exempler = int(centers[i])
        dist.append(scipy.spatial.distance.euclidean(g.node[dictionary[exempler]]['vector'], g.node[dictionary[other_exempler]]['vector']))
    return dist

def nearbyCluster(dis):
    for i in range(1,len(dis)):
        if(len(doc_clusters[dis.index(sorted(dis)[i])]) < 5):
            continue
        else:
            return dis.index(sorted(dis)[i])
            break


# In[36]:


docLabels_1 = [f for f in listdir("/Users/AR/Desktop/naturesCall/TOI_Data/actor1/") if f.endswith('.txt')]
docLabels_2 = [f for f in listdir("/Users/AR/Desktop/naturesCall/TOI_Data/actor2/") if f.endswith('.txt')]

docLabels = list(itertools.chain(docLabels_1,docLabels_2))

doc_tag = gensim.models.doc2vec.TaggedDocument
sentence_tag = gensim.models.doc2vec.TaggedLineDocument

data = []
i = 0
for doc in docLabels_1:
    file = open('/Users/AR/Desktop/naturesCall/TOI_Data/actor1/' + doc, 'r')
    data.append(doc_tag(file.read().lower().split(), [docLabels_1[i]]))
    i = i+1

j = 0
for doc in docLabels_2:
    file = open('/Users/AR/Desktop/naturesCall/TOI_Data/actor2/' + doc, 'r')
    data.append(doc_tag(file.read().lower().split(), [docLabels_2[j]]))
    j = j+1

model= Doc2Vec(data,size=25, window=5, alpha=0.025, min_alpha=0.025)

for epoch in range(10):
    model.alpha -= 0.002 
    model.min_alpha = model.alpha 
    model.train(data,total_examples=model.corpus_count,epochs=model.iter)
## model.docvecs.offset2doctag ## to view documents tags


# In[70]:


len(data)


# In[71]:


actor_1_csv = pd.read_csv('/Users/AR/Desktop/naturesCall/TOI_Data/actor1.csv')
actor_2_csv = pd.read_csv('/Users/AR/Desktop/naturesCall/TOI_Data/actor1.csv')


# In[ ]:


keywords = []
for i in range(actor_1_csv.shape[0]):
    url = actor_1_csv.iloc[i]['url']
    article = Article(url, MAX_KEYWORDS = 10)   
    article.download()

    article.parse()

    article.nlp()
    
    keywords.append(article.keywords)
    
for i in range(actor_2_csv.shape[0]):
    url = actor_1_csv.iloc[i]['url']
    article = Article(url, MAX_KEYWORDS = 10)   
    article.download()

    article.parse()

    article.nlp()
    
    keywords.append(article.keywords)


# In[37]:


g = nx.DiGraph()

h = nx.Graph()


# In[38]:


for i in range(1040):
    g.add_node(i, actor=docLabels[i][11:17], vector=model.docvecs.doctag_syn0[i], date=docLabels[i][:10], keyword = keywords[i])
    h.add_node(i, actor=docLabels[i][11:17], vector=model.docvecs.doctag_syn0[i], date=docLabels[i][:10], keyword = keywords[i])
    if(i>=1):
        for j in list(g.node):
            if((j!=i)&(g.node[i]['actor'] == g.node[j]['actor'])):  
                dist = scipy.spatial.distance.euclidean(g.node[i]['vector'], g.node[j]['vector'])
                if(dist<8):
                    g.remove_node(i)
                    break


# In[39]:


dictionary = {}
for i in range(len(g.node)):
    dictionary[i] = list(g.node)[i]
inv_dict = {v: k for k, v in dictionary.items()}


# In[40]:


X = [g.node[i]['vector'] for i in (list(g.node))] ## to convert into numpy array

# Compute Affinity Propagation
af = AffinityPropagation().fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)
    
print(n_clusters_)


# In[41]:


doc_clusters = {i: np.where(labels == i)[0] for i in range(n_clusters_)}


# In[42]:


doc_clusters


# In[43]:


new_labels = labels
min_cluster_size = 5
new_cluster_indices = cluster_centers_indices


# In[44]:


nearby_exempler = []
for i in range(len(cluster_centers_indices)):
    if(len(doc_clusters[i]) < min_cluster_size):
        exempler = int(cluster_centers_indices[i])
        distance = nearestExempler(exempler, cluster_centers_indices)
        nearby_exempler.append([i, nearbyCluster(distance)])      


# In[45]:


doc_clusters[nearby_exempler[0][0]]


# In[46]:


for i in range(len(nearby_exempler)):
    documentsforRename = doc_clusters[nearby_exempler[i][0]]
    new_cluster_indices[nearby_exempler[i][0]] = cluster_centers_indices[nearby_exempler[i][1]]
    for j in range(len(documentsforRename)):
        new_labels[documentsforRename[j]] = labels[cluster_centers_indices[nearby_exempler[i][1]]]


# In[47]:


new_cluster_indices_list = list(set(new_cluster_indices))

new_labels_list = list(set(new_labels))

doc_newclusters = {i: np.where(new_labels == i)[0] for i in new_labels_list}


# In[48]:


for i in g.node:
    g.node[i]['label'] = new_labels[inv_dict[i]]


# In[49]:


tareek = []
for i in doc_newclusters:
    cluster_date = []
    for j in doc_newclusters[i]:
        cluster_date.append(g.node[dictionary[j]]['date'])
    tareek.append(cluster_date)
#sorted_tareek = sorted(list(set(tareek)))


# In[50]:


count = 0
dict_newcluster_list = {}
for i in doc_newclusters:
    dict_newcluster_list[i] = count
    count += 1
#dict_newcluster_list = {i:j for j,i in dict_newcluster_list.items()}


# In[51]:


sort_cluster_tareek = sorted(set(tareek[dict_newcluster_list[50]]))


# In[ ]:


sort_cluster_tareek


# In[ ]:


doc_newclusters


# In[58]:


#cluster_sortdate_mapping = []
for i in doc_newclusters:
    cluster_tareek = tareek[dict_newcluster_list[i]]
    sort_cluster_tareek = sorted(set(tareek[dict_newcluster_list[i]]))
    #datewise_dict = {}
    for j in doc_newclusters[i]:
        for k in range(len(sort_cluster_tareek)):
            if(g.node[dictionary[j]]['date']==sort_cluster_tareek[k]):
                g.node[dictionary[j]]['withincluster_sortdate_label'] = k
              


# In[59]:


for i in doc_newclusters:
    #print(i)
    j = 0
    while(j+1 < len(doc_newclusters[i])):
        for k in range(j+1,len(doc_newclusters[i])):
            if(g.node[dictionary[doc_newclusters[i][j]]]['withincluster_sortdate_label']==g.node[dictionary[doc_newclusters[i][k]]]['withincluster_sortdate_label']):
                g.add_edge(dictionary[doc_newclusters[i][j]],dictionary[doc_newclusters[i][k]])
                g.add_edge(dictionary[doc_newclusters[i][k]],dictionary[doc_newclusters[i][j]])
            else:
                if(g.node[dictionary[doc_newclusters[i][j]]]['withincluster_sortdate_label']<g.node[dictionary[doc_newclusters[i][k]]]['withincluster_sortdate_label']):
                    g.add_edge(dictionary[doc_newclusters[i][j]],dictionary[doc_newclusters[i][k]])
                else:
                    g.add_edge(dictionary[doc_newclusters[i][k]],dictionary[doc_newclusters[i][j]])
        j = j+1


# In[ ]:


for i in doc_newclusters:
    for j in len(doc_newclusters[i]):
        for k in range(j+1,len(doc_newclusters)):
            if(g.node[dictionary[doc_newclusters[i][j]]]['withincluster_sortdate_label']==g.node[dictionary[doc_newclusters[i][k]]]['withincluster_sortdate_label']):
                g.add_edge(dictionary[doc_newclusters[i][j]],dictionary[doc_newclusters[i][k]])
                g.add_edge(dictionary[doc_newclusters[i][k]],dictionary[doc_newclusters[i][j]])
            else:
                if(g.node[dictionary[doc_newclusters[i][j]]]['withincluster_sortdate_label']<g.node[dictionary[doc_newclusters[i][k]]]['withincluster_sortdate_label']):
                    g.add_edge(dictionary[doc_newclusters[i][j]],dictionary[doc_newclusters[i][k]])
                else:
                    g.add_edge(dictionary[doc_newclusters[i][k]],dictionary[doc_newclusters[i][j]])
                    
            


# In[69]:


for i in doc_newclusters:
    #print(i)
    cluster_tareek = tareek[dict_newcluster_list[i]]
    sort_cluster_tareek = sorted(set(tareek[dict_newcluster_list[i]]))
    start_node = []
    end_node = []
    for j in doc_newclusters[i]:
        if(g.node[dictionary[j]]['withincluster_sortdate_label'] == 0):
            start_node.append(j)
        if(g.node[dictionary[j]]['withincluster_sortdate_label'] == len(sort_cluster_tareek)-1):
            end_node.append(j)
            
    for k in range(len(start_node)):
        for m in range(len(end_node)):
            paths = list(nx.all_simple_paths(g, source = dictionary[start_node[k]], target = dictionary[end_node[m]]))
            keywords_list = []
            for n in range(len(paths)):
                for q in range(len(paths[n])):
                    keywords_list.append(g.node[dictionary[paths[n][q]]['keyword']])
            df = pd.DataFrame(keywords_list)
            #df.to_csv('/Users/AR/Desktop/'+str(g.node[dictionary[paths[n][q]]['date']])+'_path_'+str(q), sep = ',')                 
            #df = df.append(pd.DataFrame(list, columns=['col1','col2']),ignore_index=True)
                    
            


# In[48]:


g.node[dictionary[402]]


# In[62]:


g.node[dictionary[1]]

