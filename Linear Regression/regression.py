import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data=pd.read_csv("./student/student-mat.csv",sep=";")

data=data[["G1","G2","G3","studytime","failures","absences"]]
predict="G3"
X=np.array(data.drop([predict],1)) 
y=np.array(data[predict])
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)
#Running over and over to get the best result
# best=0
# for i in range(100):
#     x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)
#     linear=linear_model.LinearRegression()
#     linear.fit(x_train,y_train)
#     accuracy=linear.score(x_test,y_test)
#     print(accuracy)
#     if accuracy>best:
#         with open("studentmodel.pickle","wb") as f:
#             pickle.dump(linear,f)

pickle_in=open("studentmodel.pickle","rb")
linear=pickle.load(pickle_in)
print("Co ",linear.coef_)
print("Intercept",linear.intercept_)
predictions=linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x]) 

#data visualization using an graph (scatter plot)   
 
style.use("ggplot")
p='studytime'
pyplot.scatter(data[p],data['G3'])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()
