import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy as sp
import itertools

columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

def LinearRegression(dataTrain, dataTest, Residue, Plot = False, PrintStr = "MSE = "):

	if Plot:
		print "Please review and close all the Histogram Plots to continue with the analysis."
		print ""
		for c in dataTrain.columns:
			if (c != 'MEDV'):
				dataTrain.hist(column = c, bins = 10)

		plt.show()

		pearsonCoeff = []
		pValues = []
		for c in dataTrain.columns:
			if (c != 'MEDV'):
				coeff,pval = sp.stats.pearsonr(dataTrain[c], dataTrain['MEDV'])
				pearsonCoeff.append([c, coeff])
				pValues.append(pval)

		print "Pearson Correlation Coefficients For all the features against the Target value (MEDV): "
		print pd.DataFrame(pearsonCoeff, columns = ['Feature', 'Pearson Coefficients'])
		print ""

	normDataTrain = (dataTrain - dataTrain.mean())/dataTrain.std()
	normDataTrain.MEDV = dataTrain.MEDV

	normDataTest = (dataTest - dataTrain.mean())/dataTrain.std()
	normDataTest.MEDV = dataTest.MEDV

	Y = np.array(dataTrain.MEDV.values).transpose()

	normDataTrain.insert(0, 'Ones', 1)
	X_Tilde = np.array(normDataTrain.drop(['MEDV'],1))
	normDataTrain = normDataTrain.drop(['Ones'],1)

	W = np.dot(X_Tilde.transpose(), X_Tilde)
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

	
	if Residue == 1:
		print PrintStr + str(MSE)
		return YVals
	else:
		return MSE

def performLinearRegression(dataTrain,dataTest):

	print "LINEAR REGRESSION:"
	print "------------------"

	MSE = LinearRegression(dataTrain,dataTrain,0,True)
	print "MSE for Training vs Training data = " + str(MSE)
	MSE = LinearRegression(dataTrain,dataTest,0)
	print "MSE for Training vs Test data = " + str(MSE)

def featureSelectionWithCorrelationHighest(dataTrain, dataTest):

	print "SELECTION WITH CORRELATION(4 highest Correlated Features):"
	print "----------------------------------------------------------"

	pearsonCoeff = []
	pValues = []
	for c in dataTrain.columns:
		if (c != 'MEDV'):
			coeff,pval = sp.stats.pearsonr(dataTrain[c], dataTrain['MEDV'])
			pearsonCoeff.append([c,abs(coeff)])
			pValues.append(pval)

	PearsonCoefficients = pd.DataFrame(pearsonCoeff, columns = ['Feature', 'PearsonCoeff'])
	PearsonCoefficients = PearsonCoefficients.sort_values(by = 'PearsonCoeff', ascending = False)

	dropColumns = []
	for x in xrange(4,len(PearsonCoefficients)):
		dropColumns.append(PearsonCoefficients.iloc[x]['Feature'])

	dataTrain = dataTrain.drop(dropColumns, 1)
	dataTest = dataTest.drop(dropColumns, 1)

	selectedFeatures = list(dataTrain.columns)
	selectedFeatures.remove('MEDV')
	print "The Four Highest correlated features selected : " + str(selectedFeatures)

	MSE = LinearRegression(dataTrain, dataTrain, 0)
	print "MSE for Training vs Training data = " + str(MSE)
	MSE = LinearRegression(dataTrain, dataTest, 0)
	print "MSE for Training vs Test data = " + str(MSE)


def featureSelectionWithCorrelationResidual(dataTrain, dataTest):

	print "SELECTION WITH CORRELATION(Residue Based):"
	print "------------------------------------------"
	
	dropColumns = list(columns)
	residue = pd.DataFrame(dataTrain.MEDV.values, columns = ['Residue'])

	for i in xrange(0,4):
		pearsonCoeff = []
		pValues = []
		for c in dropColumns:		
			coeff,pval = sp.stats.pearsonr(dataTrain[c], residue.Residue)
			pearsonCoeff.append([c,abs(coeff)])
			pValues.append(pval)


		PearsonCoefficients = pd.DataFrame(pearsonCoeff, columns = ['Feature', 'PearsonCoeff'])
		PearsonCoefficients = PearsonCoefficients.sort_values(by = 'PearsonCoeff', ascending = False)

		dropColumns = []
		for x in xrange(1,len(PearsonCoefficients)):
			dropColumns.append(PearsonCoefficients.iloc[x]['Feature'])

		dataTrainNew = dataTrain.drop(dropColumns, 1)
		dataTestNew = dataTest.drop(dropColumns, 1)

		selectedFeatures = list(dataTrainNew.columns)
		selectedFeatures.remove('MEDV')
		print "Selected Features : " + str(selectedFeatures)

		result = LinearRegression(dataTrainNew, dataTrainNew, 1 , False, "MSE for Training vs Training data = ")
		residue = result.yTrue - result.yPred
		result = LinearRegression(dataTrainNew, dataTestNew, 1, False, "MSE for Training vs Test data = ")

		print ""
		residue = pd.DataFrame(residue, columns = ['Residue'])
		

def bruteForceSearch(dataTrain, dataTest):
	
	print "BRUTE FORCE SEARCH:"
	print "-------------------"

	columnCombinations = list(itertools.combinations(columns, 4))

	dropColumns = []
	bruteForceMSETrain = []
	bruteForceMSETest = []
	for x in xrange(len(columnCombinations)):
		
		dropColumns = list(columns)
		map(lambda y:dropColumns.remove(y), list(columnCombinations[x]))

		dataTrainNew = dataTrain.drop(dropColumns, 1)
		dataTestNew = dataTest.drop(dropColumns, 1)

		result = LinearRegression(dataTrainNew, dataTrainNew, 0)
		bruteForceMSETrain.append(result)
		result = LinearRegression(dataTrainNew, dataTestNew, 0)
		bruteForceMSETest.append(result)

	print "Selected best features based on brute force search: " + str(list(columnCombinations[bruteForceMSETrain.index(min(bruteForceMSETrain))]))
	print "MSE for Training vs Training data = " + str(min(bruteForceMSETrain))
	print "MSE for Training vs Test data = " + str(bruteForceMSETest[bruteForceMSETrain.index(min(bruteForceMSETrain))])


def featureExpansion(dataTrain, dataTest):

	print "FEATURE EXPANSION:"
	print "------------------"
	
	dataTrainNew = pd.DataFrame(dataTrain)
	dataTestNew = pd.DataFrame(dataTest)

	normDataTrain = (dataTrain - dataTrain.mean())/dataTrain.std()
	normDataTrain.MEDV = dataTrain.MEDV

	normDataTest = (dataTest - dataTrain.mean())/dataTrain.std()
	normDataTest.MEDV = dataTest.MEDV

	for i in xrange(len(columns)):
		for j in xrange(i,len(columns)):
			st = str(columns[i]) + "*" + str(columns[j])
			dataTrainNew[st] = normDataTrain[columns[i]] * normDataTrain[columns[j]]
			dataTestNew[st] = normDataTest[columns[i]] * normDataTest[columns[j]]

	MSE = LinearRegression(dataTrainNew, dataTrainNew, 0)
	print "MSE for Training vs Training data = " + str(MSE)
	MSE = LinearRegression(dataTrainNew, dataTestNew, 0)
	print "MSE for Training vs Test data = " + str(MSE)

def MainLR(dataTrain, dataTest, Selection = 3):

	if Selection == 1 or Selection == 3:
		performLinearRegression(dataTrain, dataTest)
		print ""
	if Selection == 2 or Selection == 3:
		featureSelectionWithCorrelationHighest(dataTrain, dataTest)
		print ""
		featureSelectionWithCorrelationResidual(dataTrain, dataTest)
		bruteForceSearch(dataTrain, dataTest)
		print ""
		featureExpansion(dataTrain, dataTest)
	

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

	MainLR(dataTrain, dataTest)	

	