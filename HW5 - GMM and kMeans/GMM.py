import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

colors = ['ro', 'bo', 'go', 'yo', 'ko']
colorsAlt = ['r*-', 'b*-', 'g*-', 'y*-', 'k*-']

def EMAlgo(blobPoints, numberOfIterations, index):	
	k = 3
	prior = [1./k] * k
	mu = []

	for x in xrange(k):
		mu.append(np.random.randint(len(blobPoints)))

	for x in xrange(k):
		mu[x] = blobPoints.iloc[mu[x]]

	covariance = []
	for x in xrange(k):
		tmpMat = np.array(blobPoints - mu[x])
		tmpMat = np.dot(tmpMat.transpose(), tmpMat)/len(blobPoints)
		covariance.append(pd.DataFrame(tmpMat))

	conditionalProb = np.zeros((len(blobPoints),k))
	posterior = np.zeros((len(blobPoints),k))
	likelihoodPlot = []

	for i in xrange(numberOfIterations):
		newPrior = [0.0] * k
		newMu = [0.0] * k

		logLikelihood = 0.0
		for x in xrange(len(blobPoints)):
			for y in xrange(k):	
				conditionalProb[x][y] = multivariate_normal.pdf(np.array(blobPoints.iloc[x]), np.array(mu[y]),np.array(covariance[y]))
			total = 0.0
			for y in xrange(k):	
				total += conditionalProb[x][y] * prior[y]
			for y in xrange(k):	
				posterior[x][y] = conditionalProb[x][y] * prior[y] / total
				newMu[y] += posterior[x][y] * blobPoints.iloc[x]
				newPrior[y] += posterior[x][y]
			logLikelihood += np.log(total)

		for y in xrange(k):
			newMu[y] /= newPrior[y]
			newPrior[y]/=len(blobPoints)

		mu = newMu
		prior = newPrior
		
		newCovariance = []
		for y in xrange(k):	
			sumMat = 0.0
			for x in xrange(len(blobPoints)):
				tmpMat = np.matrix(blobPoints.iloc[x] - mu[y])
				tmpMat = np.dot(tmpMat.transpose(), tmpMat)
				tmpMat *= posterior[x][y]
				sumMat += tmpMat
			sumMat /= newPrior[y] * len(blobPoints)
			newCovariance.append(pd.DataFrame(sumMat))

		covariance = newCovariance

		likelihoodPlot.append([i + 1, logLikelihood])

	likelihoodPlot = pd.DataFrame(likelihoodPlot, columns = ['Iterations', 'LogLikelihood'])
	plt.title('Loglikelihood for various runs')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Log Likelihood')
	plt.plot(likelihoodPlot.Iterations,likelihoodPlot.LogLikelihood, colorsAlt[index])

	return mu, covariance, conditionalProb, likelihoodPlot.LogLikelihood.max()
	
	
def GMMMain():
	print "GMM Algorithm"
	print "---------------"

	blobPoints = pd.read_csv('hw5_blob.csv', header = None)
	blobPoints.columns = ['X', 'Y']
	k = 3
	noOfTimes = 5
	numberOfIterations = 50

	muList = []
	covarianceList = []
	conditionalProbList = []
	logLikelihoodList = []

	for x in xrange(noOfTimes):
		print "GMM for " + str(numberOfIterations) + " Iterations - Attempt" + str(x+1) + " : "
		print "----------------------------------"
		mu, covariance, conditionalProb, logLikelihood = EMAlgo(blobPoints,numberOfIterations, x)
		muList.append(mu)
		covarianceList.append(covariance)
		conditionalProbList.append(conditionalProb)
		logLikelihoodList.append(logLikelihood)

	#print "Plot of Log Likelihood"

	maxLikelihood = max(logLikelihoodList)
	maxIndex = logLikelihoodList.index(maxLikelihood)

	mu = muList[maxIndex]
	covariance = covarianceList[maxIndex]
	conditionalProb = conditionalProbList[maxIndex]
	
 	plt.figure()
	print "Best Values Selected"
	print "---------------------"
	print "Log Likelihood = " + str(maxLikelihood)
	for y in xrange(k):
		print "\nFor Cluseter " + str(y+1) + ": " 
		print "----------------" 
		print "Mean"
		print "------"
		print str(np.matrix(mu[y]))
		print "Covariance"
		print "-----------"
		print str(np.matrix(covariance[y]))

	print "Plotting the required cluster plot"
	plt.title('GMM Cluster Plot')
	for x in xrange(len(blobPoints)):
		condProb = list(conditionalProb[x])
		maxProb = max(condProb)
		maxIndex = condProb.index(maxProb)
		plt.plot(blobPoints.X[x],blobPoints.Y[x], colors[maxIndex])
	print "Close the Plots to continue execution"
	plt.show()

if __name__ == '__main__':
	GMMMain()
