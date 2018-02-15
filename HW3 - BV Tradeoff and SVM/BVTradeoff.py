import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp

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

	return (MSE,W)


def LinearRegression(dataTrain, dataTest, Residue, Plot = False, PrintStr = "MSE = "):

	
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

	return (MSE, W)


def BiasVarianceTradeoff(Samples, Ridge = False,lambdaVal = 0):
	DataSets = []
	for x in xrange(100):
		DataSets.append(-1 + 2 * np.random.rand(Samples))

	ySet = []

	for x in xrange(100):
		yTemp = []
		for y in xrange(Samples):
			yTemp.append(2 * (DataSets[x][y]* DataSets[x][y]) + np.random.normal(0, np.sqrt(0.1)))
		ySet.append(yTemp)

	g1_MSE = []
	g2_MSE = []
	g3_MSE = []
	g4_MSE = []
	g5_MSE = []
	g6_MSE = []
	
	g2_W = []
	g3_W = []
	g4_W = []
	g5_W = []
	g6_W = []

	for x in xrange(100):
		
		#G1
		MSE = ((ySet[x] - np.ones(Samples)) ** 2).mean()
		g1_MSE.append(MSE)

		#G2
		Data = pd.DataFrame(ySet[x], columns=['MEDV'])
		if Ridge:
			MSE, W_g2 = LinearRegression(Data,Data,0)
		else:
			MSE, W_g2 = RidgeRegression(Data,Data,lambdaVal)
		g2_MSE.append(MSE) 
		g2_W.append(W_g2)

		#G3
		Data['X'] = DataSets[x]
		if Ridge:
			MSE, W_g3 = LinearRegression(Data,Data,0)
		else:
			MSE, W_g3 = RidgeRegression(Data,Data,lambdaVal)
		g3_MSE.append(MSE)
		g3_W.append(W_g3)

		Data['X_2'] = DataSets[x] ** 2
		if Ridge:
			MSE, W_g4 = LinearRegression(Data,Data,0)
		else:
			MSE, W_g4 = RidgeRegression(Data,Data,lambdaVal)
		g4_MSE.append(MSE)
		g4_W.append(W_g4)

		Data['X_3'] = DataSets[x] ** 3
		if Ridge:
			MSE, W_g5 = LinearRegression(Data,Data,0)
		else:
			MSE, W_g5 = RidgeRegression(Data,Data,lambdaVal)
		g5_MSE.append(MSE)
		g5_W.append(W_g5)

		Data['X_4'] = DataSets[x] ** 4
		if Ridge:
			MSE, W_g6 = LinearRegression(Data,Data,0)
		else:
			MSE, W_g6 = RidgeRegression(Data,Data,lambdaVal)
		g6_MSE.append(MSE)
		g6_W.append(W_g6)


	if Ridge == False:
		print "Plotting Histograms for all the MSE values of G functions when NUmber of Samples = " + str(Samples)
		pd.DataFrame(g1_MSE,columns = ['G1_MSE']).hist()
		pd.DataFrame(g2_MSE,columns = ['G2_MSE']).hist()
		pd.DataFrame(g3_MSE,columns = ['G3_MSE']).hist()
		pd.DataFrame(g4_MSE,columns = ['G4_MSE']).hist()
		pd.DataFrame(g5_MSE,columns = ['G5_MSE']).hist()
		pd.DataFrame(g6_MSE,columns = ['G6_MSE']).hist()
	plt.show()

	g2_W = pd.DataFrame(g2_W, columns = ['w0'])
	g3_W = pd.DataFrame(g3_W, columns = ['w0','w1'])
	g4_W = pd.DataFrame(g4_W, columns = ['w0','w1','w2'])
	g5_W = pd.DataFrame(g5_W, columns = ['w0','w1','w2','w3'])
	g6_W = pd.DataFrame(g6_W, columns = ['w0','w1','w2','w3','w4'])


	TestDataLength = 100
	DataSets = []

	for x in xrange(TestDataLength):
		DataSets.append(-1 + 2 * np.random.rand(Samples))

	ySet = []

	for x in xrange(TestDataLength):
		yTemp = []
		for y in xrange(Samples):
			yTemp.append(2 * (DataSets[x][y] * DataSets[x][y]) + np.random.normal(0, np.sqrt(0.1)))
		ySet.append(yTemp)

	bias1 = []
	bias2 = []
	bias3 = []
	bias4 = []
	bias5 = []
	bias6 = []
	var1 = []
	var2 = []
	var3 = []
	var4 = []
	var5 = []
	var6 = []

	for x in xrange(TestDataLength):
		p_y_x = sp.norm.pdf(ySet[x], 2 * (DataSets[x] ** 2), np.sqrt(0.1))	
		prob = 1./Samples * p_y_x

		Eh1 = np.ones(Samples) 
		bias1.append(sum(((Eh1-ySet[x])**2) * prob))
		tmpVar = 0
		tmpVar = sum(((Eh1-(sum(Eh1)/len(Eh1))) ** 2 ) * prob)
		var1.append(tmpVar/100)

		Data = pd.DataFrame(np.ones(Samples),columns = ['Ones'])
		tmp = np.array(Data)
		Eh2 = g2_W.w0.mean() * np.ones(Samples)
		bias2.append(sum(((Eh2-ySet[x])**2) * prob))
		tmpVar = 0
		for y in xrange(100):		
			h2 = np.dot(np.array(g2_W.iloc[y]),tmp.transpose())
			tmpVar += sum(((h2-Eh2) ** 2 ) * prob)
		var2.append(tmpVar/100)

		Data['X'] = DataSets[x]
		tmp = np.array(Data)
		Eh3 = np.dot(np.array(g3_W.mean()),tmp.transpose())
		bias3.append(sum(((Eh3-ySet[x])**2) * prob))
		tmpVar = 0
		for y in xrange(100):		
			h3 = np.dot(np.array(g3_W.iloc[y]),tmp.transpose())
			tmpVar += sum(((h3-Eh3) ** 2 ) * prob)
		var3.append(tmpVar/100)

		Data['X_2'] = DataSets[x] ** 2
		tmp = np.array(Data)
		Eh4 = np.dot(np.array(g4_W.mean()),tmp.transpose())
		bias4.append(sum(((Eh4-ySet[x])**2) * prob))
		tmpVar = 0
		for y in xrange(100):		
			h4 = np.dot(np.array(g4_W.iloc[y]),tmp.transpose())
			tmpVar += sum(((h4-Eh4) ** 2 ) * prob)
		var4.append(tmpVar/100)
		
		Data['X_3'] = DataSets[x] ** 3
		tmp = np.array(Data)
		Eh5 = np.dot(np.array(g5_W.mean()),tmp.transpose())
		bias5.append(sum(((Eh5-ySet[x])**2) * prob))
		tmpVar = 0
		for y in xrange(100):		
			h5 = np.dot(np.array(g5_W.iloc[y]),tmp.transpose())
			tmpVar += sum(((h5-Eh5) ** 2 ) * prob)
		var5.append(tmpVar/100)
		
		Data['X_4'] = DataSets[x] ** 4
		tmp = np.array(Data)
		Eh6 = np.dot(np.array(g6_W.mean()),tmp.transpose())
		bias6.append(sum(((Eh6-ySet[x])**2) * prob))
		tmpVar = 0
		for y in xrange(100):		
			h6 = np.dot(np.array(g6_W.iloc[y]),tmp.transpose())
			tmpVar += sum(((h6-Eh6) ** 2 ) * prob)
		var6.append(tmpVar/100)
		
	Finalbias1 = sum(bias1)/len(bias1)
	Finalbias2 = sum(bias2)/len(bias2)
	Finalbias3 = sum(bias3)/len(bias3)
	Finalbias4 = sum(bias4)/len(bias4)
	Finalbias5 = sum(bias5)/len(bias5)
	Finalbias6 = sum(bias6)/len(bias6)

	FinalVar1 = sum(var1)/len(var1)
	FinalVar2 = sum(var2)/len(var2)
	FinalVar3 = sum(var3)/len(var3)
	FinalVar4 = sum(var4)/len(var4)
	FinalVar5 = sum(var5)/len(var5)
	FinalVar6 = sum(var6)/len(var6)

	if Ridge == False:
		print "BiasSquare for g1 = " + str(Finalbias1)
		print "BiasSquare for g2 = " + str(Finalbias2)
		print "BiasSquare for g3 = " + str(Finalbias3)
		print "BiasSquare for g4 = " + str(Finalbias4)
		print "BiasSquare for g5 = " + str(Finalbias5)
		print "BiasSquare for g6 = " + str(Finalbias6)

		print "Variance for g1 = " + str(FinalVar1)
		print "Variance for g2 = " + str(FinalVar2)
		print "Variance for g3 = " + str(FinalVar3)
		print "Variance for g4 = " + str(FinalVar4)
		print "Variance for g5 = " + str(FinalVar5)
		print "Variance for g6 = " + str(FinalVar6)
	else:
		print "BiasSquare for g4 = " + str(Finalbias4)
		print "Variance for g4 = " + str(FinalVar4)

def MainBVT():
	print"Bias Variance Tradeoff"
	print"-----------------------\n"

	print "Linear Regression with 10 samples"
	print "----------------------------------"
	BiasVarianceTradeoff(10)
	print ""

	print "Linear Regression with 100 samples"
	print "-----------------------------------"
	BiasVarianceTradeoff(100)
	print ""

	lambdaValues = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

	for lambdaVal in lambdaValues:
		print "Ridge Regression with 100 samples and Lambda = " + str(lambdaVal)	
		print "-----------------------------------------------------"	
		BiasVarianceTradeoff(100,True,lambdaVal)
		print""

if __name__ == '__main__':
	MainBVT()

