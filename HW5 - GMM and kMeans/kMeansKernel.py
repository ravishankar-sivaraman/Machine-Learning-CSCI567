import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def KernelKMeans(dataPoints, k, title):
	c = []

	dataPoints['X2'] = dataPoints.X ** 2 + dataPoints.Y ** 2

	for x in xrange(k):
		c.append(np.random.randint(len(dataPoints)))
	
	for x in xrange(k):
		c[x] = dataPoints.iloc[c[x]]

	l2Dist = []

	newL2Dist = []
	while True:
		for x in xrange(k):
			l2Dist.append((dataPoints - c[x]) **2)
			l2Dist[x]['Dist'] = l2Dist[x]['X2']# + l2Dist[x]['XY'] + l2Dist[x]['Y2']

		newL2Dist = []
		for x in xrange(k):
			newL2Dist.append([])
		for i in xrange(len(dataPoints)):	
			dist = []
			for x in xrange(k):
				dist.append(l2Dist[x].Dist[i])

			minDist = min(dist)
			minIndex = dist.index(minDist)

			newL2Dist[minIndex].append(dataPoints.iloc[i])

		for x in xrange(k):
			newL2Dist[x] = pd.DataFrame(newL2Dist[x])

		newC = [0] * k
		for x in xrange(k):
			newC[x] = newL2Dist[x].mean()

		equalFlag = True
		for x in xrange(k):
			if newC[x].X != c[x].X or newC[x].Y != c[x].Y:
				equalFlag = False
				break
		if equalFlag:
			break

		c = newC

	colors = ['ro', 'bo', 'yo', 'go', 'ko']

	plt.title(title)
	for x in xrange(k):
		plt.plot(newL2Dist[x].X,newL2Dist[x].Y, colors[x])


def kMKernelMain():
	print "Kernel K Means Algorithm"
	print "--------------------------"

	circlePoints = pd.read_csv('hw5_circle.csv', header = None)
	circlePoints.columns = ['X', 'Y']

	print "Plotting the required cluster plots"
	KernelKMeans(circlePoints, 2, "Kernel K Means for hw5_circle.csv. K = 2")
	print "Close the Plots to continue execution"
	
	plt.show()

if __name__ == '__main__':
	kMKernelMain()

