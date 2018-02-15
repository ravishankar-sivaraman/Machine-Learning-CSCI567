import hw_utils as hw
from timeit import default_timer

xTrain,yTrain,xTest,yTest = hw.loaddata('MiniBooNE_PID.txt')
xTrainNorm, xTestNorm = hw.normalize(xTrain,xTest)

dIn = 50
dOut = 2

xTrain = xTrainNorm
yTrain = yTrain
xTest = xTestNorm
yTest = yTest

print "\nLinear Activations"
print "-------------------"
print "Architecture 1"
print "---------------"
architectures = [[dIn, dOut], [dIn, 50, dOut], [dIn, 50, 50, dOut], [dIn, 50, 50, 50, dOut]]
startTime = default_timer()
hw.testmodels(xTrain, yTrain, xTest, yTest, architectures, 'linear', 'softmax', [0.0], 30, 1000, 0.001, [0.0], [0.0], False, False, 0)
timeTaken = default_timer() - startTime
print "Training time = " + str(timeTaken) + " s"

print "Architecture 2"
print "---------------"
architectures = [[dIn, 50, dOut], [dIn, 500, dOut], [dIn, 500, 300, dOut], [dIn, 800, 500, 300, dOut], [dIn, 800, 800, 500, 300, dOut]]
startTime = default_timer()
hw.testmodels(xTrain, yTrain, xTest, yTest, architectures, 'linear', 'softmax', [0.0], 30, 1000, 0.001, [0.0], [0.0], False, False, 0)
timeTaken = default_timer() - startTime
print "Training time = " + str(timeTaken) + " s"

print "\nSigmoid Activations"
print "--------------------"
architectures = [[dIn, 50, dOut], [dIn, 500, dOut], [dIn, 500, 300, dOut], [dIn, 800, 500, 300, dOut], [dIn, 800, 800, 500, 300, dOut]]
startTime = default_timer()
hw.testmodels(xTrain, yTrain, xTest, yTest, architectures, 'sigmoid', 'softmax', [0.0], 30, 1000, 0.001, [0.0], [0.0], False, False, 0)
timeTaken = default_timer() - startTime
print "Training time = " + str(timeTaken) + " s"

print "\nReLu Activations"
print "-----------------"
architectures = [[dIn, 50, dOut], [dIn, 500, dOut], [dIn, 500, 300, dOut], [dIn, 800, 500, 300, dOut], [dIn, 800, 800, 500, 300, dOut]]
startTime = default_timer()
hw.testmodels(xTrain, yTrain, xTest, yTest, architectures, 'relu', 'softmax', [0.0], 30, 1000, 5e-4, [0.0], [0.0], False, False, 0)
timeTaken = default_timer() - startTime
print "Training time = " + str(timeTaken) + " s"

print "\nReLu Activations with L2 Regularization"
print "----------------------------------------"
architectures = [[dIn, 800, 500, 300, dOut]]
regCoeffs=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
startTime = default_timer()
bestRegular = hw.testmodels(xTrain, yTrain, xTest, yTest, architectures, 'relu', 'softmax', regCoeffs, 30, 1000, 5e-4, [0.0], [0.0], False, False, 0)
timeTaken = default_timer() - startTime
print "Training time = " + str(timeTaken) + " s"

print "\nReLu Activations with L2 Regularization and Early Stopping"
print "-----------------------------------------------------------"
architectures = [[dIn, 800, 500, 300, dOut]]
regCoeffs=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
startTime = default_timer()
bestRegularES = hw.testmodels(xTrain, yTrain, xTest, yTest, architectures, 'relu', 'softmax', regCoeffs, 30, 1000, 5e-4, [0.0], [0.0], False, True, 0)
timeTaken = default_timer() - startTime
print "Training time = " + str(timeTaken) + " s"

print "\nSGD with weight decay"
print "----------------------"
architectures = [[dIn, 800, 500, 300, dOut]]
regCoeffs=[5e-7]
sgdDecays=[1e-5, 5e-5, 1e-4, 3e-4, 7e-4, 1e-3]
startTime = default_timer()
bestDecay = hw.testmodels(xTrain, yTrain, xTest, yTest, architectures, 'relu', 'softmax', regCoeffs, 100, 1000, 1e-5, sgdDecays, [0.0], False, False, 0)
timeTaken = default_timer() - startTime
print "Training time = " + str(timeTaken) + " s"

print "\nMomentum"
print "---------"
architectures = [[dIn, 800, 500, 300, dOut]]
sgdMoms = [0.99, 0.98, 0.95, 0.9, 0.85]
startTime = default_timer()
bestMomentum = hw.testmodels(xTrain, yTrain, xTest, yTest, architectures, 'relu', 'softmax', [0.0], 50, 1000, 1e-5, [bestDecay[2]], sgdMoms, True, False, 0)
timeTaken = default_timer() - startTime
print "Training time = " + str(timeTaken) + " s"

print "\nCombining the above"
print "--------------------"
architectures = [[dIn, 800, 500, 300, dOut]]
bestRegularization = []
if(bestRegular[5] >= bestRegularES[5]):
	bestRegularization.append(bestRegular[1])
else:
	bestRegularization.append(bestRegularES[1])
startTime = default_timer()
hw.testmodels(xTrain, yTrain, xTest, yTest, architectures, 'relu', 'softmax', bestRegularization, 100, 1000, 1e-5, [bestDecay[2]], [bestMomentum[3]], True, True, 0)
timeTaken = default_timer() - startTime
print "Training time = " + str(timeTaken) + " s"

print "\nGrid search with cross validation"
print "----------------------------------"
architectures = [[dIn, 50, dOut], [dIn, 500, dOut], [dIn, 500, 300, dOut], [dIn, 800, 500, 300, dOut], [dIn, 800, 800, 500, 300, dOut]]
regCoeffs=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
sgdDecays=[1e-5, 5e-5, 1e-4]
startTime = default_timer()
hw.testmodels(xTrain, yTrain, xTest, yTest, architectures, 'relu', 'softmax', regCoeffs, 100, 1000, 1e-5, sgdDecays, [0.99], True, True, 0)
timeTaken = default_timer() - startTime
print "Training time = " + str(timeTaken) + " s"
