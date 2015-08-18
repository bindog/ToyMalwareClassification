from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


subtrainLabel = pd.read_csv('subtrainLabels.csv')
subtrainfeature = pd.read_csv("imgfeature.csv")
subtrain = pd.merge(subtrainLabel,subtrainfeature,on='Id')
labels = subtrain.Class
subtrain.drop(["Class","Id"], axis=1, inplace=True)
subtrain = subtrain.as_matrix()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(subtrain,labels,test_size=0.4)

srf = RF(n_estimators=500, n_jobs=-1)
srf.fit(X_train,y_train)
print srf.score(X_test,y_test)

# importances = srf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in srf.estimators_],axis=0)
# indices = np.argsort(importances)[::-1]
# print("Feature ranking:")
# for f in range(20):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
