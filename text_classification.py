# this code is for text classification

from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans



categories = [
'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x'
]


newsgroup_train = fetch_20newsgroups(subset='train', categories= categories)
newsgroup_test = fetch_20newsgroups(subset='test', categories= categories)


# check if the dataset has been full load
# pprint(list(pprint(newsgroup_train.target_names)))

vectorizer = HashingVectorizer(stop_words='english', non_negative=True, n_features=10000)
fea_train = vectorizer.fit_transform(newsgroup_train.data)
fea_test = vectorizer.fit_transform(newsgroup_test.data)

# check train and test data's feature vector
# print('Size of fea_train:' + repr(fea_train.shape))
# print('Size of fea_test:' + repr(fea_test.shape))

#calculate the none zero elements in the train data
# print('the average feature sparsity is {0:.3f}%'.format(fea_train.nnz/(fea_train.shape[0] * fea_train.shape[1]) * 100))

def calculate_result(actual, pred):
    m_precision = metrics.precision_score(actual, pred,average=None)
    m_recall = metrics.recall_score(actual, pred,average=None)
    print('predict info:')

    print(type(m_precision))
    print(np.mean(m_precision,dtype=np.float32))
    print(np.mean(m_recall, dtype=np.float32))
    print(np.mean(metrics.f1_score(actual, pred,average=None),dtype=np.float32))
    # print('precision:{0:.3f}'.format(m_precision))
    # print('recall:{0:0.3f}'.format(m_recall))
    # print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, pred,average=None)))


# NB classification
print("--------------NB--------------")
clf = MultinomialNB(alpha=0.01)
clf.fit(fea_train, newsgroup_train.target)
pred = clf.predict(fea_test)
calculate_result(newsgroup_test.target, pred)

# KNN classification
print("--------------KNN--------------")
knnclf = KNeighborsClassifier()
knnclf.fit(fea_train, newsgroup_train.target)
pred = knnclf.predict(fea_test)
calculate_result(newsgroup_test.target, pred)


# SVM classification
print("--------------SVM--------------")
svcclf = SVC(kernel='linear')
svcclf.fit(fea_train, newsgroup_train.target)
pred = svcclf.predict(fea_test)
calculate_result(newsgroup_test.target, pred)


# Kmeans cluster
print("--------------Kmeans--------------")
pred = KMeans(n_clusters=5)
pred.fit(fea_test)
calculate_result(newsgroup_test.target, pred.labels_)