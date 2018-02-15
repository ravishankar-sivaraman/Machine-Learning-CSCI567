import numpy as np
import pandas as pd
import LinearRegression as LR
import RidgeRegression as RR

columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

if __name__ == '__main__':

	from sklearn.datasets import load_boston
	boston = load_boston()

	data = pd.DataFrame(boston.data)
	target = pd.DataFrame(boston.target)

	data.columns = columns
	target.columns = ['MEDV']

	data['MEDV'] = target

	dataTrain = data.iloc[lambda data: data.index%7 != 0].reset_index()
	dataTest = data.iloc[lambda data: data.index%7 == 0].reset_index()

	dataTrain = dataTrain.drop(['index'],1)
	dataTest = dataTest.drop(['index'],1)

	LR.MainLR(dataTrain, dataTest, 1)

	RR.MainRR(dataTrain, dataTest)
	print ""
	LR.MainLR(dataTrain, dataTest, 2)