import numpy as np
import pandas as pd

def tieBreak(Lst1, Lst2, dist):
	jointList = pd.merge(Lst1,Lst2, on=['Class'], how = 'inner')
	jointList.columns = ['ID_X', 'ActualClass_X', 'Class', 'L1_X', 'L2_X', 'ID_Y','ActualClass_Y', 'L1_Y', 'L2_Y']
	col = ['Column','L1_X', 'L2_X']
	jointList = jointList[jointList[col[dist]] == jointList[col[dist]].min()]
	return list(jointList['Class'])

def ClassifyKNN(LList, dist):
	L = pd.DataFrame(LList);
	L.columns = ['ID', 'ActualClass', 'Class', 'L1', 'L2']
	Lnew = L.groupby('Class').count().reset_index()
	Lnew = Lnew[Lnew['L1'] == Lnew['L1'].max()]
	clas = list(Lnew['Class'])
	if len(Lnew) > 1:
		clas = tieBreak(L, Lnew, dist)	
	return clas[0]

def getAccuracyKNN(train,test, LeaveOneOut = False):
	df = pd.read_csv(train, header = None)
	dfTest = pd.read_csv(test, header = None)

	df.columns = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe','Type']
	dfTest.columns = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe','Type']

	normDf = (df - df.mean())/df.std()
	normDf.ID = df.ID
	normDf.Type = df.Type

	normTestDf = (dfTest - df.mean())/df.std()
	normTestDf.ID = dfTest.ID
	normTestDf.Type = dfTest.Type

	# dfList = list(df.values.tolist())
	# dfTestList = list(dfTest.values.tolist())

	dfList = list(normDf.values.tolist())
	dfTestList = list(normTestDf.values.tolist())

	classificationL1 = []
	classificationL2 = []

	for k in [1,3,5,7]:
		classificationL1 = []
		classificationL2 = []
		print "For k = " + str(k)
		for x in xrange(0,len(dfTestList)):
			distances = []
			for y in xrange(0,len(dfList)):
				if(LeaveOneOut and dfTestList[x][0] == dfList[y][0]):
					continue
				L1dist = 0
				L2dist = 0
				for z in xrange(1,len(dfTestList[0]) - 1):
					L1dist += abs((dfList[y][z] - dfTestList[x][z]))
					L2dist += (dfList[y][z] - dfTestList[x][z]) ** 2
				L2dist = np.sqrt(L2dist)

				distances.append([dfList[y][0], dfTestList[x][10], dfList[y][10], L1dist, L2dist]) #ID, Class, L1, L2

			distancesL1 = sorted(distances, key=lambda a_entry: a_entry[3])
			distancesL2 = sorted(distances, key=lambda a_entry: a_entry[4])
			
			classificationL1.append([dfTestList[x][0], dfTestList[x][10], ClassifyKNN(distancesL1[:k],1)])
			classificationL2.append([dfTestList[x][0], dfTestList[x][10], ClassifyKNN(distancesL2[:k],2)])

		l1Class = pd.DataFrame(classificationL1)
		l2Class = pd.DataFrame(classificationL2)
		l1Class.columns = ['ID', 'ActualClass', 'AssignedClass']
		l2Class.columns = ['ID', 'ActualClass', 'AssignedClass']

		print "L1 Accuracy = " + str(l1Class[l1Class['ActualClass'] == l1Class['AssignedClass']].count()['ID'].astype(float)/len(dfTest)*100)
		print "L2 Accuracy = " + str(l2Class[l2Class['ActualClass'] == l2Class['AssignedClass']].count()['ID'].astype(float)/len(dfTest)*100)

def kNNMain():
	print "k - NEAREST NEIGHBORS:"
	print "----------------------"
	print "Testing with Training Data:"
	getAccuracyKNN('train.txt','train.txt', True)
	print ""
	print "Testing with Test Data:"
	getAccuracyKNN('train.txt','test.txt')

if __name__ == '__main__':
	kNNMain()
