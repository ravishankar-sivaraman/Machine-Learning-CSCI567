import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def SimpleKMeans(dataPoints, k, title):
	c = []

	for x in xrange(k):
		c.append(np.random.randint(len(dataPoints)))

	for x in xrange(k):
		c[x] = dataPoints.iloc[c[x]]

	l2Dist = []

	newL2Dist = []
	while True:
		for x in xrange(k):
			l2Dist.append((dataPoints - c[x]) **2)
			l2Dist[x]['Dist'] = l2Dist[x]['X'] + l2Dist[x]['Y']

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

	plt.figure()
	plt.title(title)
	for x in xrange(k):
		plt.plot(newL2Dist[x].X,newL2Dist[x].Y, colors[x])
	#plt.show()

def kMMain():
	print "K Means Algorithm"
	print "-------------------"

	circlePoints = pd.read_csv('hw5_circle.csv', header = None)
	blobPoints = pd.read_csv('hw5_blob.csv', header = None)

	circlePoints.columns = ['X', 'Y']
	blobPoints.columns = ['X', 'Y']

	print "Plotting the required cluster plots"
	SimpleKMeans(blobPoints, 2, "K Means for hw5_blob.csv. K = 2")
	SimpleKMeans(blobPoints, 3, "K Means for hw5_blob.csv. K = 3")
	SimpleKMeans(blobPoints, 5, "K Means for hw5_blob.csv. K = 5")

	SimpleKMeans(circlePoints, 2, "K Means for hw5_circle.csv. K = 2")
	SimpleKMeans(circlePoints, 3, "K Means for hw5_circle.csv. K = 3")
	SimpleKMeans(circlePoints, 5, "K Means for hw5_circle.csv. K = 5")

	print "Close the Plots to continue execution"
	plt.show()

if __name__ == '__main__':
	kMMain()
