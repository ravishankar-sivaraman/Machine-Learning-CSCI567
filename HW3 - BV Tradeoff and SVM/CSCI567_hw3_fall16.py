import numpy as np
import pandas as pd
import BVTradeoff as BVT
import libsvm as LI
import LinearKernelSVM as SVM

columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

if __name__ == '__main__':
	BVT.MainBVT()
	SVM.MainSVM()
	LI.MainLIBSVM()
