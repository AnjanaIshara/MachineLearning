import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
cancer= datasets.load_breast_cancer()
print(cancer.feature_names)
print(cancer.target_names)

X=cancer.data
y=cancer.target
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2)

classes=['malignant','benign']
# Hyperplane must be generated accoring to the largest distance of the two data points
#which allows to maximize the margin
# deciding a hyperplane is difficult when the datapoints doesnot show clear seperation
#using kernel the above problem can be rectified 
#softmargin can be used to allow for outliers

clf=svm.SVC(kernel='linear',C=2)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
acc=metrics.accuracy_score(y_test,y_pred)
print(acc)





