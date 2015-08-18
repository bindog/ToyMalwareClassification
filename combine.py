from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import pandas as pd


subtrainLabel = pd.read_csv('subtrainLabels.csv')
subtrainfeature1 = pd.read_csv("3gramfeature.csv")
subtrainfeature2 = pd.read_csv("imgfeature.csv")
subtrain = pd.merge(subtrainfeature1,subtrainfeature2,on='Id')
subtrain = pd.merge(subtrain,subtrainLabel,on='Id')
labels = subtrain.Class
subtrain.drop(["Class","Id"], axis=1, inplace=True)
subtrain = subtrain.as_matrix()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(subtrain,labels,test_size=0.4)

srf = RF(n_estimators=500, n_jobs=-1)
srf.fit(X_train,y_train)
print srf.score(X_test,y_test)