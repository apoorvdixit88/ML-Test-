import matplotlib.pyplot as plt
import numpy as np
from sklearn import  datasets,linear_model
from sklearn.metrics import mean_squared_error

dia=datasets.load_diabetes()
dia_x=dia.data
dia_x_train=dia_x[:-40]
dia_x_test=dia_x[-20:]
dia_y_train=dia.target[:-40]
dia_y_test=dia.target[-20:]
model=linear_model.LinearRegression()
model.fit(dia_x_train,dia_y_train)
dia_y_predict=model.predict(dia_x_test)
print("mean sq error is",mean_squared_error(dia_y_test,dia_y_predict))
#print(dia_x)
#print(dia.DESCR)
#(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename',
#dia_x=dia.data(:,np.axis)
print("weights and intercept",model.coef_,model.intercept_)
# plt.scatter(dia_x_test,dia_y_test)
# plt.plot(dia_x_test,dia_y_predict)
# plt.show()

