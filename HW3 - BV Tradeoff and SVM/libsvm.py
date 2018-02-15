from svmutil import *
import scipy.io as sio
import numpy as np
import pandas as pd
from timeit import default_timer
import LinearKernelSVM as lsvm

def MainLIBSVM():

	print "\nRunning LIBSVM"
	print "---------------"
	dataTrain = sio.loadmat('phishing-train.mat')
	dataTest = sio.loadmat('phishing-test.mat')

	features =  pd.DataFrame(dataTrain['features'])
	featuresTest =  pd.DataFrame(dataTest['features'])

	labels = pd.DataFrame(dataTrain['label'])
	labelsTest = pd.DataFrame(dataTest['label'])

	featuresNew = lsvm.TransformData(features)
	featuresNewTest = lsvm.TransformData(featuresTest)

	best = None
	fileExists = True
	try:
		best = pd.DataFrame.from_csv('BestParams.csv')
	except IOError as e:
		fileExists = False
		

	y = dataTrain['label'].transpose()
	x = np.array(featuresNew)
	x = list(x.tolist())
	prob  = svm_problem(y , x)


	if fileExists == False:
		resultsL = lsvm.LinearSVM(prob)
		resultsP = lsvm.PolynomialKernelSVM(prob)
		resultsR = lsvm.RBFKernelSVM(prob)
		best = lsvm.getMax(resultsL, resultsP, resultsR)	
		best.to_csv('BestParams.csv')

	best = best.iloc[0]
	best = best.tolist()


	yTest = dataTest['label'].transpose()
	xTest = np.array(featuresNewTest)
	xTest= list(xTest.tolist())
	probTest  = svm_problem(yTest , xTest)

	print "Running SVM for Test Data with best parameters"
	print "-----------------------------------------------"
	paramStr = '-t 2 -c ' + str(best[0]) + ' -q' + ' -g ' + str(best[1])
	print "Best Params set as : \"" + paramStr + "\""
	param = svm_parameter(paramStr)
	m = svm_train(prob, param)


	paramStr = ''
	p_labels, p_acc, p_vals = svm_predict(yTest, xTest, m, '')

if __name__ == '__main__':
	MainLIBSVM()