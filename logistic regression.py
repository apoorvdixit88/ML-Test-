"""algorithm which does classification
by calculating probability belonging to a particular"""
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import  pyplot as plt
iris=datasets.load_iris()
# print(list(iris.keys()))
# print(iris['data'].shape)
# print(iris['target_names'])
# print(iris['DESCR'])
#taking data of column 2
x=iris['data'][:,3:]
# print(x)
#true for virginica
y=(iris['target']==2).astype(np.int)
# print(y)
#trainig logic regress classifie
clf=LogisticRegression()
clf.fit(x,y)
ex=clf.predict([[2.6],[1.5]])
print(ex)
#using matplot lib
x_n=np.linspace(0,3,1000).reshape(-1,1)
# print(x_n)....
x_p=clf.predict_proba(x_n)
plt.plot(x_n,x_p[:,1],"g-",label="virginica")
plt.show()


