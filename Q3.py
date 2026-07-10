import numpy as np

data = np.loadtxt("Assignment1_ML_Data/Hogsmeade_prices.csv", skiprows=1, delimiter=',')

x = data[:,:-1]
y = data[:,-1]

nFeatures = x.shape[1]

print(f"Number of features: {nFeatures}")
print(f"Number of samples: {x.shape[0]}")
print(f"Price range: {y.min()} to {y.max()}")

featureNames = [f"Feature {i+1}" for i in range(nFeatures)]

xMean = np.mean(x, axis=0)
xStd = np.std(x, axis=0)
xStd[xStd == 0] = 1
xNorm = (x - xMean) / xStd

np.random.seed(13)
nSamples = len(x)
indinces = np.random.permutation(nSamples)
trainSize = int(0.8 * nSamples)

trainIDX = indinces[:trainSize]
testIDX = indinces[trainSize:]

xTrain = xNorm[trainIDX]
xTest = xNorm[testIDX]
yTrain = y[trainIDX]
yTest = y[testIDX]

print(f"Training samples: {len(xTrain)}")
print(f"Testing samples: {len(xTest)}")

class LinearRegression:

    def __init__(self): 
        self.theta = None

    def add_intercept(self, x):
        nSamples = x.shape[0]
        intercept = np.hstack((np.ones((nSamples, 1)), x))
        return intercept

    def fit(self, x, y):

        intercept = self.add_intercept(x)
        y = y.flatten()
        XtX = np.dot(intercept.T, intercept)
        XtX_inverse = np.linalg.inv(XtX)
        XtY = np.dot(intercept.T, y)
        self.theta = np.dot(XtX_inverse, XtY)

    def predict(self, x):
        intercept = self.add_intercept(x)
        return np.dot(intercept, self.theta)
    
    def getEQ(self, featureNames = None):

        if self.theta is None:
            return "Model is not trained yet."
        
        eq = f"y = {self.theta[0]:.4f}"
        for i in range(1, len(self.theta)):
            if featureNames and i-1 < len(featureNames):
                feature = featureNames[i-1]
            else:
                feature = f"x{i}"
            eq += f" + {self.theta[i]:.4f} * {feature}"
        return eq
    
    def getFeatureImport(self):
        if self.theta is None:
            return None
        return np.abs(self.theta[1:])
    
model = LinearRegression()
model.fit(xTrain, yTrain)

yHatTrain = model.predict(xTrain)
yHatTest = model.predict(xTest)

trainMSE = np.mean((yHatTrain - yTrain.flatten()) ** 2)
testMSE = np.mean((yHatTest - yTest.flatten()) ** 2)

print("Simple Linear Regression Results:")
print(f"Training MSE: {trainMSE:.4f}")
print(f"Testing MSE: {testMSE:.4f}")
print(f"Learned Equation: {model.getEQ(featureNames)}")

importances = model.getFeatureImport()

print("feature importances:")
print("most impact on price:")

featureImport=[]
for i in range(len(featureNames)):
    featureImport.append((featureNames[i], importances[i]))

featureImport.sort(key=lambda x: x[1], reverse=True)

for rank, (feature, importance) in enumerate(featureImport, 1):
    print(f"{rank}. {feature} (Importance: {importance:.4f})")

mostImportantFeature = featureImport[0][0]
secoondMostImportantFeature = featureImport[1][0] if len(featureImport) > 1 else "N/A"

print(f"Factor with most impact on price: {mostImportantFeature}")

