import numpy as np
import matplotlib.pyplot as plt


x_train = np.loadtxt("Assignment1_ML_Data/x_train.csv", skiprows=1)
y_train = np.loadtxt("Assignment1_ML_Data/y_train.csv", skiprows=1)
x_test = np.loadtxt("Assignment1_ML_Data/x_test.csv", skiprows=1)
y_test = np.loadtxt("Assignment1_ML_Data/y_test.csv", skiprows=1)

plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, color="blue", label="Training Data")
plt.title("Training Data Distribution")
plt.xlabel("x_train")
plt.ylabel("y_train")

plt.subplot(1, 2, 2)
plt.scatter(x_test, y_test, color="red", label="Testing Data")
plt.title("Testing Data Distribution")
plt.xlabel("x_test")
plt.ylabel("y_test")
plt.legend()

plt.tight_layout()
plt.show()

def create_polynomial(x, degree):

    nSamples = x.shape[0] 
    xPoly = np.zeros((nSamples, degree)) 

    for i in range(degree):
        xPoly[:, i] = x.flatten() ** (i + 1) 
    return xPoly



class LinearRegression:

    def __init__(self, learningRate = .01, n = 1000): 
        self.learningRate = learningRate
        self.n = n
        self.weight = None
        self.bias = None
        self.losses = []

    def fit(self, x, y): 

        nSamples, nFeatures = x.shape
        self.weight = np.zeros(nFeatures)
        self.bias = 0

        for i in range(self.n):
            yHat = np.dot(x, self.weight) + self.bias
            errors = yHat - y

            dw = (1/nSamples) * np.dot(x.T, errors)
            db = (1/nSamples) * np.sum(errors)

            self.weight -= self.learningRate * dw
            self.bias -= self.learningRate * db

            loss = np.mean(errors ** 2)
            self.losses.append(loss)

            if i % 100 == 0: 
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict(self, x): 
        return np.dot(x, self.weight) + self.bias
    
    def getEQ(self, degree): 
        eq = f"y = {self.bias:.4f}"

        for i in range(degree):
            eq += f" + {self.weight[i]:.4f} * x^{i+1}"
        
        return eq
    

degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
bestDegree = 0
bestLoss = float('inf')
bestModel = None

plt.figure(figsize=(15, 10))

for idx, degree in enumerate(degrees):
    print(f"\n{'='*50} ")
    print(f"Training Polynomial Regression with Degree {degree}")
    print("="*50)
    
    xTrainPoly = create_polynomial(x_train, degree)
    xTestPoly = create_polynomial(x_test, degree)

    mean = np.mean(xTrainPoly, axis=0)
    std = np.std(xTrainPoly, axis=0)
    std[std == 0] = 1 

    xTrainPoly = (xTrainPoly - mean) / std
    xTestPoly = (xTestPoly - mean) / std

    model = LinearRegression(learningRate=0.01, n=1000)
    model.fit(xTrainPoly, y_train)

    yHatTrain = model.predict(xTrainPoly)
    yHatTest = model.predict(xTestPoly)

    trainLoss = np.mean((yHatTrain - y_train) ** 2)
    testLoss = np.mean((yHatTest - y_test) ** 2)

    print(f"\nResults for Degree {degree}:")
    print(f"Training Loss: {trainLoss:.4f}")
    print(f"Testing Loss: {testLoss:.4f}")
    print(f"Learned Equation: {model.getEQ(degree)}")

    plt.subplot(2, 5, idx + 1)

    plt.scatter(x_train, y_train, alpha=.5, label = "Training Data (y)")
    plt.scatter(x_test, y_test, alpha=.5, label = "Test Data (y)")

    x_min = min(x_train.min(), x_test.min()) 
    x_max = max(x_train.max(), x_test.max())  
    x_smooth = np.linspace(x_min, x_max, 300).reshape(-1, 1)
    x_smooth_poly = create_polynomial(x_smooth, degree)
    x_smooth_poly = (x_smooth_poly - mean) / std
    y_smooth = model.predict(x_smooth_poly)
    plt.plot(x_smooth, y_smooth, "r-", linewidth=2, label=f"Degree {degree} Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Degree{degree} Polynomial Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if testLoss < bestLoss:
        bestLoss = testLoss
        bestDegree = degree
        bestModel = model

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
plt.plot(bestModel.losses, label="Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title(f"Training Loss Curve for Best Model (Degree {bestDegree})")
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n{'='*50} ")
print(f"Best Model: Degree {bestDegree}")
print("="*50)
print(f"Best test loss: {bestLoss:.4f}")
print(f"Best Model EQ: {bestModel.getEQ(bestDegree)}")

