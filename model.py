import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle

df = pd.read_csv("winequalityN.csv")

mean = df["fixed acidity"].mean()
df["fixed acidity"].fillna(mean,inplace=True)

mean2 = df["volatile acidity"].mean()
df["volatile acidity"].fillna(mean,inplace=True)

mean3 = df["citric acid"].mean()
df["citric acid"].fillna(mean,inplace=True)

mean4 = df["residual sugar"].mean()
df["residual sugar"].fillna(mean,inplace=True)

mean4 = df["chlorides"].mean()
df["chlorides"].fillna(mean,inplace=True)

mean5 = df["pH"].mean()
df["pH"].fillna(mean,inplace=True)

mean6 = df["sulphates"].mean()
df["sulphates"].fillna(mean,inplace=True)

lower_limit = df["free sulfur dioxide"].mean() - 3*df["free sulfur dioxide"].std()
upper_limit = df["free sulfur dioxide"].mean() + 3*df["free sulfur dioxide"].std()

df2 = df[(df["free sulfur dioxide"] > lower_limit) & (df["free sulfur dioxide"] < upper_limit)]

lower_limit = df2['total sulfur dioxide'].mean() - 3*df2['total sulfur dioxide'].std()
upper_limit = df2['total sulfur dioxide'].mean() + 3*df2['total sulfur dioxide'].std()

df3 = df2[(df2['total sulfur dioxide'] > lower_limit) & (df2['total sulfur dioxide'] < upper_limit)]

lower_limit = df3['residual sugar'].mean() - 3*df3['residual sugar'].std()
upper_limit = df3['residual sugar'].mean() + 3*df3['residual sugar'].std()

df4 = df3[(df3['residual sugar'] > lower_limit) & (df3['residual sugar'] < upper_limit)]

dummies = pd.get_dummies(df4["type"],drop_first=True)

df4 = pd.concat([df4,dummies],axis=1)

df4.drop("type",axis=1,inplace=True)

quaity_mapping = { 3 : "Low",4 : "Low",5: "Medium",6 : "Medium",7: "Medium",8 : "High",9 : "High"}
df4["quality"] =  df4["quality"].map(quaity_mapping)

mapping_quality = {"Low" : 0,"Medium": 1,"High" : 2}
df4["quality"] =  df4["quality"].map(mapping_quality)

x = df4.drop("quality",axis=True)
y = df4["quality"]

clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf.fit(x,y)

pickle.dump(clf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
#
print(model.predict([[8.3,0.21,0.58,17.1,0.049,62,213,1.006,3.01,0.51,9.3,1]]))
