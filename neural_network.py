import numpy as np

# fully flexible multi layer classifier neural network

class classifier():
    def __init__(self, shape):
        self.shape = shape
        self.inputs = shape[0]
        self.outputs = shape[-1]

        self.weights = []
        self.biases = []

        for i in range(len(self.shape)-1):
            self.biases.append(np.random.random((self.shape[i+1], 1)))
            self.weights.append(np.random.random((self.shape[i+1], self.shape[i])))

        self.learningRate = 1
        self.epochs = 100
  
    def train(self, dataset):
        costs = []
        for iter in range(self.epochs):
            n = len(self.shape)-1
            for ind in range(n):
                self.weights[ind] -= self.learningRate * self.costDerivative(dataset, 1, ind)
                self.biases[ind] -= self.learningRate * self.costDerivative(dataset, 0, ind)

            cost = self.costFunction(dataset)
            # here
            cost = cost[0]
            costs.append(cost)
            print('Epoch :', iter+1, '; Cost :', cost)
        return np.array(costs)

    def costDerivative(self, dataset, p, ind):
        n = len(self.shape)-1
        derivative = np.zeros(self.weights[ind].shape if p else self.biases[ind].shape)
        for datapoint in dataset:
            a = self.zedFunction(datapoint[0], n)
            # a = self.tanhFunction(z)
            y = datapoint[1]
            d = 2*(a-y) # cost derivative

            for i in range(n, ind+1, -1):
                a = self.zedFunction(datapoint[0], i)
                # a = self.tanhFunction(z)
                d *= a*(1-a)
                # d = 1 - d**2
                d = np.transpose(d)
                d = np.matmul(d, self.weights[i-1])
                d = np.transpose(d)

            a = self.zedFunction(datapoint[0], ind+1)
            # a = self.tanhFunction(z)
            d *= a*(1-a)
            # d = 1-d**2
            x = self.zedFunction(datapoint[0], ind)
            x = np.transpose(x)
            derivative += np.matmul(d, x) if p else d # weight derivative

        derivative /= len(dataset)
        return derivative

    def costFunction(self, dataset):
        n = len(self.shape)-1
        error = 0
        for datapoint in dataset:
            a = self.zedFunction(datapoint[0], n)
            # a = self.tanhFunction(z)
            y = datapoint[1]
            c = y-a
            c **= 2
            # cost = (c[0] + c[1])/self.outputs # for two outputs
            cost = c**(1/2) # for one output
            error += cost
        error /= len(dataset)
        return error**(1/2)

    def sigmoidFunction(self, z):
        return np.exp(z)/(np.exp(z)+1)

    def tanhFunction(self, z):
        return np.tanh(z)

    def zedFunction(self, datapoint, index):
        a = datapoint
        for i in range(index):
            a = np.matmul(self.weights[i], a) + self.biases[i]
            a = self.sigmoidFunction(a)
        return a
