import numpy as np
import pandas as pd
import math

def ClassifyNB(LList):
	L = pd.DataFrame(LList);
	L.columns = ['Class', 'Likelihood']
	Lnew = L[L['Likelihood'] == L['Likelihood'].max()]
	clas = list(Lnew['Class'])
	
	return clas[0]

def NormPDF(x, mean, variance):
	if variance != 0:
		expr1 = 1.0/(math.sqrt(2 * math.pi * variance))
		expr2 = math.exp((-1.0 / (2 * variance)) * ((x - mean) ** 2))
		return expr1 * expr2
	else:
		return 1
def getAccuracyNB(train, test):
	df = pd.read_csv(train, header = None)
	dfTest = pd.read_csv(test, header = None)

	df.columns = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe','Class']
	dfTest.columns = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe','Class']


	priors = df.groupby('Class').count().reset_index().drop(['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'],1)
	priors.columns = ['Class','Prior']

	priors.Prior /= priors.Prior.sum()

	meanCl = []
	varCl = []
	for x in list(df.Class.unique()):
		meanAttr = []
		varAttr = []
		meanAttr.append(df.loc[x]['ID'])
		varAttr.append(df.loc[x]['ID'])
		for y in df.columns:
			if y!='ID' and y!='Class':
				meanAttr.append(df[df['Class'] == x][y].mean())
				varAttr.append(df[df['Class'] == x][y].var())
		meanAttr.append(x)
		varAttr.append(x)
		meanCl.append(meanAttr)
		varCl.append(varAttr)

	meanDF = pd.DataFrame(meanCl)
	varDF = pd.DataFrame(varCl)
	meanDF.columns = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe','Class']
	varDF.columns = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe','Class']

	classification = []

	for x in xrange(0,len(dfTest)):
		likelihood = []
	 	for y in xrange(0,len(meanDF)):
	 		prob = np.log(priors.loc[y][1])
	 		for z in df.columns:
	 			if z!='ID' and z!='Class':
	 				prob += np.log(NormPDF(dfTest.loc[x][z], meanDF.loc[y][z], varDF.loc[y][z]))
	 		likelihood.append([meanDF.loc[y][10],prob])
	 	
	 	classification.append([dfTest.loc[x][0], dfTest.loc[x][10], ClassifyNB(likelihood)])

	outClass = pd.DataFrame(classification)
	outClass.columns = ['ID', 'ActualClass', 'AssignedClass']

	print "Accuracy = " + str(outClass[outClass['ActualClass'] == outClass['AssignedClass']].count()['ID'].astype(float)/len(dfTest)*100)

def NBMain():
	print "NAIVE BAYES:"
	print "------------"
	print "Testing with Training Data:"
	getAccuracyNB('train.txt','train.txt')
	print ""
	print "Testing with Test Data:"
	getAccuracyNB('train.txt','test.txt')

if __name__ == '__main__':
	NBMain()




