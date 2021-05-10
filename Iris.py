#Analysis of Iris Dataset
# Machine Learning project
# H Hurchand @ 28 November 2019
# Continuing Education Program - Concordia University

# The program uses supervised learning algorithms to test performance near boundaries of overlapping data.
# For the Iris dataset this occurs for the Viginica and Versicolor samples

# The program also uses PCA to conduct a dimension reduction
# It tests the algorithms used above on the pca data set. A performance graph is illustrated.

# Data
from sklearn.datasets import load_iris
from sklearn import preprocessing
import pandas as pd

# For Visualisation
#
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Estimators
# Deterministic Estimators

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ML Libraries
from  sklearn.model_selection import train_test_split

# Validation
from sklearn.metrics import confusion_matrix,accuracy_score

iris = load_iris()

unscaled_iris_df = pd.DataFrame(iris.data,columns=iris.feature_names)


#getFeatures = unscaled_iris_df.columns
# Scale data
scaler = preprocessing.StandardScaler()

iris_df = scaler.fit_transform(unscaled_iris_df)
iris_df = pd.DataFrame(iris_df,columns=iris.feature_names)

X = iris_df
y = iris.target

correl = pd.DataFrame(iris_df.groupby(y).corr())
sns.set(style="white")
#print(correl.to_string())
sns.heatmap(correl,annot=True)

# HyperParameter tuning for KNN
#knnDict ={}
# for param in range(1,100,2):
#     kfold=KFold(n_splits=30,random_state=11,shuffle=True)
#     knn = KNeighborsClassifier(n_neighbors=param)
#     scores = cross_val_score(knn,X,y,cv=kfold)
#     k1 = {param:[scores.mean(),scores.std()]}
#     knnDict.update(k1)

# dataX = pd.DataFrame.from_dict(knnDict,orient='index')
# #print(dataX)
# dataX.columns=["mean accuracy","error"]
# ax = sns.lineplot(data=dataX,x=dataX.index.values,y=dataX["mean accuracy"])
# ax = plt.axes()
# ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
# plt.show()

ModelDict = {"GaussianNB":GaussianNB(),"KNN":KNeighborsClassifier(),"SVC rbf":SVC(kernel='rbf',gamma='auto')
             ,"SVC poly":SVC(kernel='poly',degree=5,gamma='auto'),"SVC linear":SVC(kernel='linear',gamma='auto')}

metricDict={"GaussianNB":[],"KNN":[],"SVC rbf":[],"SVC poly":[],"SVC linear":[]}
metricDictPCA={"GaussianNB":[],"KNN":[],"SVC rbf":[],"SVC poly":[],"SVC linear":[]}

# Dimension reduction using PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=2,random_state=11)
pca.fit(iris_df)
PCA(copy=True,iterated_power='auto',n_components=2,random_state=11,svd_solver='auto',tol=0.0,whiten=False)
iris_pca=pca.transform(iris_df)
#print(iris_pca.shape)
print("Exaplained variance of two pca Components",pca.explained_variance_)
iris_pca_df=pd.DataFrame(iris_pca,columns=['Component1','Component2'])
iris_pca_df['species']=iris.target

fig1, ax1 = plt.subplots()
ax1=sns.scatterplot(data=iris_pca_df,x='Component1',y='Component2',hue='species',legend='brief'
            ,palette='cool')
plt.show()

metricDict1={"GaussianNB":[],"KNN":[],"SVC rbf":[],"SVC poly":[],"SVC linear":[]}

dataSet = {"original": iris_df,"pca": iris_pca_df}
#import random as ran
for iter in range(1,51):
#    rs = ran.randint(1,100)
    for Model_name,Model in ModelDict.items():
        for i in dataSet.keys():
# Usual DataSet

            dataType = i
            data_df = dataSet[i]
            X_train, X_test, y_train, y_test = train_test_split(data_df,iris.target, test_size=0.25,shuffle=True,
                                                            random_state=10)
# PCA DataSet
            learned = Model.fit(X_train,y_train)
            predicted = learned.predict(X_test)
            A = confusion_matrix(y_test,predicted)
            accu=accuracy_score(y_test,predicted)

            if dataType == 'original':
               metricDict[Model_name].append(accu)
            else:
               metricDictPCA[Model_name].append(accu)

from statistics import mean,stdev

dataClass = {}
dataClassPCA = {}
for keys in metricDict:
         a = metricDict[keys]
         k1 = {keys: [mean(a) * 100, stdev(a) * 100]}
         dataClass.update(k1)

for keys in metricDictPCA:
         a = metricDictPCA[keys]
         k1 = {keys: [mean(a) * 100, stdev(a) * 100]}
         dataClassPCA.update(k1)

DataVal1= pd.DataFrame.from_dict(dataClass,orient='index')
DataVal2= pd.DataFrame.from_dict(dataClassPCA,orient='index')
DataVal1.columns=["Mean score","Deviation"]
DataVal2.columns=["Mean score","Deviation"]
fig2,ax2=plt.subplots()
ax2.errorbar(x=DataVal1.index,y=DataVal1["Mean score"],yerr=DataVal1["Deviation"])
ax2.errorbar(x=DataVal2.index,y=DataVal2["Mean score"],yerr=DataVal2["Deviation"])
ax2.set_xlabel('Classifiers')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy of different classifiers on 50 identical runs')
ax2.set(ylim=(70,120))
plt.show()


# Changed the last line by adding this comment
# These are changes I made from branch 1
