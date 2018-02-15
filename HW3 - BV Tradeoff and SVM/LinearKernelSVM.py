from svmutil import *
import scipy.io as sio
import numpy as np
import pandas as pd
from timeit import default_timer

def TransformData(features):		
	cols = [1, 6, 7, 13, 14, 25, 28]

	index = 0
	for c in features.columns:
		if c in cols:
			C0 = []
			C1 = []
			C2 = []
			for x in xrange(len(features[c])):
				C0.append(0) # -1
				C1.append(0) # 0 
				C2.append(0) # 1 
				if features.iloc[x][c] == 1:
					C2[x] = 1
				elif features.iloc[x][c] == 0:
					C1[x] = 1
				else:
					C0[x] = 1

			features = features.drop([c],1)
			features.insert(index,str(c)+'_0', C0)
			index+=1
			features.insert(index,str(c)+'_1', C1)
			index+=1
			features.insert(index,str(c)+'_2', C2)
		index+=1

	featuresNew = pd.DataFrame()
	for c in features.columns:
		featuresNew[c] = features[c].apply(lambda value: 1 if value == 1 else 0)

	return featuresNew

def LinearSVM(prob):
	C= [4**-6, 4**-5, 4**-4, 4**-3, 4**-2, 4**-1, 4**0, 4**1, 4**2]
	print "\nLinear SVM"
	print "-----------"
	results = []
	start = default_timer()
	for c in C:	
		print 'C = ' + str(c)
		paramStr = '-t 0 -c ' + str(c) + ' -v 3 -q'
		param = svm_parameter(paramStr)
		m = svm_train(prob, param)

		results.append([str(c), m])
	results = pd.DataFrame(results)
	avgTime = default_timer() - start
	avgTime/=27
	print "\nAverage time = " + str(avgTime)
	results = results[results[1] == results[1].max()]

	print "Best Accuracy for C = " + str(results.iloc[0][0]) + " with accuracy = " + str(results.iloc[0][1])
	return results

def PolynomialKernelSVM(prob):
	C= [4**-3, 4**-2, 4**-1, 4**0, 4**1, 4**2, 4**3, 4**4, 4**5, 4**6, 4**7]
	degree = [1,2,3]


	results = []
	print "\nPolynomial Kernel SVM"
	print "-----------------------"
	start = default_timer()
	for c in C:
		for d in degree:		
			print 'C = ' + str(c) + ' and Degree = ' + str(d)	
			paramStr = '-t 1 -c ' + str(c) + ' -v 3 -q' + ' -d ' + str(d)
			param = svm_parameter(paramStr)
			m = svm_train(prob, param)

			results.append([str(c), str(d), m])

	results = pd.DataFrame(results)
	avgTime = default_timer() - start
	avgTime/=99
	print "\nAverage time = " + str(avgTime)

	results = results[results[2] == results[2].max()]
	print "Best Accuracy for C = " + str(results.iloc[0][0]) + ", Degree = " + str(results.iloc[0][1]) + " with accuracy = " + str(results.iloc[0][2])
	return results

def RBFKernelSVM(prob):
	C= [4**-3, 4**-2, 4**-1, 4**0, 4**1, 4**2, 4**3, 4**4, 4**5, 4**6, 4**7]
	gamma = [4**-7, 4**-6, 4**-5, 4**-4, 4**-3, 4**-2, 4**-1]
	results = []
	print "\nRBF kernel SVM"
	print "----------------"
	start = default_timer()
	for c in C:
		for g in gamma:				
			print 'C = ' + str(c) + ' and Gamma = ' + str(g)
			paramStr = '-t 2 -c ' + str(c) + ' -v 3 -q' + ' -g ' + str(g)
			param = svm_parameter(paramStr)
			m = svm_train(prob, param)

			results.append([str(c), str(g), m])

	results = pd.DataFrame(results)
	avgTime = default_timer() - start
	avgTime/=231
	print "\nAverage time = " + str(avgTime)
	results = results[results[2] == results[2].max()]
	print "Best Accuracy for C = " + str(results.iloc[0][0]) + ", Gamma = " + str(results.iloc[0][1]) + " with accuracy = " + str(results.iloc[0][2])
	return results

def getMax(results1, results2, results3):
	if results1.iloc[0][1] > results2.iloc[0][2]:
		if results1.iloc[0][1] > results3.iloc[0][2]:
			return results1
		else:
			return results3
	else:
		if results2.iloc[0][2] > results3.iloc[0][2]:
			return results2
		else:
			return results3


def MainSVM():

	print "\nRunning SVM:"
	print "-------------"

	dataTrain = sio.loadmat('phishing-train.mat')
	dataTest = sio.loadmat('phishing-test.mat')

	features =  pd.DataFrame(dataTrain['features'])
	featuresTest =  pd.DataFrame(dataTest['features'])

	labels = pd.DataFrame(dataTrain['label'])
	labelsTest = pd.DataFrame(dataTest['label'])


	featuresNew = TransformData(features)
	featuresNewTest = TransformData(featuresTest)

	y = dataTrain['label'].transpose()
	x = np.array(featuresNew)
	x = list(x.tolist())
	prob  = svm_problem(y , x)

	resultsL = LinearSVM(prob)
	resultsP = PolynomialKernelSVM(prob)
	resultsR = RBFKernelSVM(prob)


	best = getMax(resultsL, resultsP, resultsR)	
	best.to_csv('BestParams.csv')
	best = best.iloc[0]
	best = best.tolist()


	yTest = dataTest['label'].transpose()
	xTest = np.array(featuresNewTest)
	xTest= list(xTest.tolist())
	probTest  = svm_problem(yTest , xTest)

	print "\nRunning for Test Data with best parameters : "
	print "---------------------------------------------"
	paramStr = '-t 2 -c ' + str(best[0]) + ' -q' + ' -g ' + str(best[1])
	print "Params = \"" + paramStr + "\""
	param = svm_parameter(paramStr)
	m = svm_train(prob, param)


	paramStr = ''
	p_labels, p_acc, p_vals = svm_predict(yTest, xTest, m, '')


if __name__ == '__main__':
	MainSVM()