import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import linear_model,preprocessing


data=pd.read_csv('car.data')
print(data.head())
pre=preprocessing.LabelEncoder()
#preprocessing assigns string valuse with numbers
buying=pre.fit_transform(list(data["buying"]))
maint=pre.fit_transform(list(data["maint"]))
doors=pre.fit_transform(list(data["door"]))
persons=pre.fit_transform(list(data["persons"]))
lug_boot=pre.fit_transform(list(data["lug_boot"]))
safety=pre.fit_transform(list(data["safety"]))
c=pre.fit_transform(list(data["class"]))

predict="class"
X=list(zip(buying,maint,doors,persons,lug_boot,safety))#zip can be used to combine preprocessed tuples into a one
y=list(c)

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)
print(x_train)
print(y_test)
#select k as an odd because the winning number of a voting should be odd
#if we use high number of k values and if the belonging data set have less than K then will output AN INCORRECT output
#practical issue is finding the distance to the each of the data poin so this is useless

model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
acc=model.score(x_test,y_test)
print(acc)

names=["unacc","acc","good","vgood"]
predicted=model.predict(x_test)
for x in range(len(x_test)):
    print("Predicted: ",names[predicted[x]],"Data: ",x_test[x],"Actual : ",names[y_test[x]])
    n=model.kneighbors([x_test[x]],9,True)
    print("N : ",n)