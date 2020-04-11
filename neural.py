import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def cost_func(a, y):
    return 0.5 * np.power(a - y, 2).mean()

def cost_derivative(a, y):
    return (a - y)

def convert_label(label, num_class=None):
    temp = np.array(label)
    if num_class == None:
        num_class = temp.max() + 1
    res = np.zeros((len(temp), num_class), dtype=int)
    for i in range(len(temp)):
        res[i][temp[i]] = 1
    return res
    
    

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(self.layers)
        self.alpha = 0.3    
    
    def fit(self, data):
        self.X_train = np.array(data[0])
        self.N = len(self.X_train)
        self.y_train = convert_label(data[1])

        self.w = [np.random.randn(l0, l1) for l0, l1 in zip(self.layers[:-1], self.layers[1:])]
        self.b = [np.random.rand(l0) for l0 in self.layers[1:]]

    def feedforward(self, X = None):
        a = [X] if X != None else[self.X_train]
        z = []
        for i in range(self.num_layers - 1):
            z_ = np.dot(a[i], self.w[i]) + self.b[i]
            a_ = sigmoid(z_)
            z.append(z_)
            a.append(a_)
        return a, z
    
    def backprop(self, a, z):
        derivative_a = [cost_derivative(a[-1], self.y_train)]
        derivative_w = []
        derivative_b = []
        derivative_z = []

        for i in range(self.num_layers - 2, -1, -1):
            derivative_z.append(derivative_a[-1] * sigmoid_prime(z[i]))
            derivative_a.append(np.dot(derivative_z[-1], self.w[i].T))
            derivative_w.append(np.dot(a[i].T, derivative_z[-1]) / self.N)
            derivative_b.append(derivative_z[-1].mean(axis=0))

        derivative_w.reverse()
        derivative_b.reverse()
        return derivative_w, derivative_b

    def GD(self, maximum):
        for i in range(maximum):
            print(i+1," time:")
            a, z = self.feedforward()
            cost = cost_func(a[-1], self.y_train)
            print("cost: ", cost)
            derivative_w, derivative_b = self.backprop(a, z)
            self.update(derivative_w, derivative_b)

    def update(self, derivative_w, derivative_b):
        self.w = [x - self.alpha * y for x, y in zip(self.w, derivative_w)]
        self.b = [x - self.alpha * y for x, y in zip(self.b, derivative_b)]
    
    def predict(self, X_test):
        res_matrix = self.feedforward(X_test)[-1]
        res = []
        for element in res_matrix:
            for i in range(len(element)):
                if element[i] == 1:
                    res += [i]
        return res

# X_train = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
# y_train = [0, 1, 0]
    
# data = (X_train, y_train)
# layers = [3, 2, 2, 2]
    
# network = NeuralNetwork(layers)
# network.fit(data)
# a, z = network.feedforward()
# derivative_w, derivative_b = network.backprop(a, z)
# network.GD(100)
# print("X_train: ", network.X_train)
# print("y_train: ", network.y_train)
# print("w: ", network.w)
# print("b: ", network.b)
# print("z: ", z)
# print("a: ", a)

# print("derivative_w: ", derivative_w)
# print("derivative_b: ", derivative_b)