import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy as sp

columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

def RidgeRegression(dataTrain, dataTest, lambdaVal):


	normDataTrain = (dataTrain - dataTrain.mean())/dataTrain.std()
	normDataTrain.MEDV = dataTrain.MEDV

	normDataTest = (dataTest - dataTrain.mean())/dataTrain.std()
	normDataTest.MEDV = dataTest.MEDV

	Y = np.array(dataTrain.MEDV.values).transpose()

	normDataTrain.insert(0, 'Ones', 1)
	X_Tilde = np.array(normDataTrain.drop(['MEDV'],1))
	normDataTrain = normDataTrain.drop(['Ones'],1)

	lambdaIMatrix = np.identity(len(X_Tilde[0])) * lambdaVal

	W = np.dot(X_Tilde.transpose(), X_Tilde)
	W = W + lambdaIMatrix
	W = np.linalg.pinv(W)
	W = np.dot (W, X_Tilde.transpose())
	W = np.dot(W, Y)

	normDataTest.insert(0, 'Ones', 1)
	X_Tilde = np.array(normDataTest.drop(['MEDV'],1))
	normDataTest = normDataTest.drop(['Ones'],1)

	YPredictions = []
	for x in xrange(len(normDataTest)):
		YPredictions.append(np.dot(W,X_Tilde[x]))

	YPredictions = np.array(YPredictions)

	YVals = pd.DataFrame(YPredictions.transpose())
	YVals.columns = ['yPred']
	YVals['yTrue'] = normDataTest['MEDV']

	MSE = ((YVals.yTrue - YVals.yPred) ** 2).mean()

	return MSE


def performRidgeRegression(dataTrain,dataTest):

	print "RIDGE REGRESSION:"
	print "-----------------"

	lambdaValList = [0.01, 0.1, 1.0]
	for lambdaVal in lambdaValList:
		print "Lambda = " + str(lambdaVal)
		MSE = RidgeRegression(dataTrain,dataTrain,lambdaVal)
		print "MSE for Training vs Training data = " + str(MSE)
		MSE = RidgeRegression(dataTrain,dataTest,lambdaVal)
		print "MSE for Training vs Test data = " + str(MSE)
		print ""

def performTenFoldsCrossValidation(dataTrain,dataTest):

	print "RIDGE REGRESSION WITH TEN-FOLDS CROSS VALIDATION:"
	print "-------------------------------------------------"

	dataTrainSplit = np.array_split(dataTrain.iloc[np.random.permutation(len(dataTrain))], 10)

	results = []
	lambdaValList = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
	for x in xrange(len(dataTrainSplit)):
		dataForTrain = pd.DataFrame()

		for y in xrange(len(dataTrainSplit)):
			if y == x:
				dataForTest = dataTrainSplit[y].reset_index().drop(['index'],1)
			else:
				dataForTrain = dataForTrain.append(dataTrainSplit[y], ignore_index = True)

		for lambdaVal in lambdaValList:
			MSE = RidgeRegression(dataForTrain, dataForTest, lambdaVal)
			results.append([x, lambdaVal, MSE])

	RidgeRegCV = pd.DataFrame(results, columns = ['SelectedBin', 'SelectedLambda', 'MSE'])
	meanLambdaMSETrain = []
	meanLambdaMSETest = []
	for lambdaVal in lambdaValList:
		meanLambdaMSETrain.append([lambdaVal, RidgeRegCV[RidgeRegCV['SelectedLambda'] == lambdaVal]['MSE'].mean()])
		meanLambdaMSETest.append([lambdaVal, RidgeRegression(dataTrain, dataTest, lambdaVal)])

	print "Cross Validated Ridge Regression MSE for Training vs Training data for different lambda Values :"
	print pd.DataFrame(meanLambdaMSETrain, columns = ['LambdaValue', 'MSE'])
	print ""
	print "Ridge Regression MSE for Training vs Test data for different lambda Values :"
	print pd.DataFrame(meanLambdaMSETest, columns = ['LambdaValue', 'MSE'])


def MainRR(dataTrain, dataTest):

	performRidgeRegression(dataTrain, dataTest)
	performTenFoldsCrossValidation(dataTrain, dataTest)
	

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

	MainRR(dataTrain, dataTest)

