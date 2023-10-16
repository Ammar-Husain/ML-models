"""
This is implementaion of Multiple variable Linear Regression model with gradient decent algorithm and mean squared error as cost function.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

class LineaRegression:
    def __init__(self, w=np.zeros(1), b=0, lr=0.5):
        self.initial_w = w # used to access the inital w even after training
        self.w = w
        self.initial_b = b # used to access the inital b even after training
        self.b = b
        self.lr = lr

    def train(self, X, Y, n_iters=500, learning_curve=False, lr=0):
        w = self.w if self.w.shape == X.shape[1] else np.zeros(X.shape[1]) # initiate w with zeros if the provided w shape is not suitable
        b = self.initial_b
        if lr == 0:
            lr = self.lr
        
        temp_w = w.copy() # for simultaneous update
        m = len(X)
        
        if learning_curve: # I could check `if learning_curve` inside the training loop but i made it like this so the check happen one time instead of n_iters times
            costs = [] # for learning curve
            for _ in range(n_iters):
                # non victorized implementation try it and compare the time complexity
                """
                for j in range(len(temp_w)):
                    temp_w[j] -= - lr * np.mean(
                        [(self.predict(X[i], w, b) - Y[i]) * X[i][j] for i in range(len(X))]
                    )
                    
                b -= lr * np.mean(
                    [(self.predict(X[i], w, b) - Y[i]) for i in range(len(X))]
                )
                """
                # victorized implementation
                temp_w -= lr * 1/m * (X@w + b - Y)@X
                b -= lr * np.mean(X@w + b - Y)
                
                w = temp_w.copy()
                costs.append(self.cost(X=X, Y=Y, w=w, b=b))
        
            # print(costs[:10]) # for debugging purposes
            plt.plot(list(range(1, n_iters + 1)), costs)
            plt.title("change in cost funtion value throug iterations")
            plt.xlabel("iteration")
            plt.ylabel("cost function value")
            
            if not os.path.exists("plots"):
               os.mkdir("plots")
            # save the figure in ./plots/multi_linear_learning_curve.png
            plt.savefig("plots/multi_linear_learning_curve.png", dpi=300)
            
        else:
            for _ in range(n_iters):
                # non victorized implementation try it and compare the time complexity
                """
                for j in range(len(temp_w)):
                    temp_w[j] -= - lr * np.mean(
                        [(self.predict(X[i], w, b) - Y[i]) * X[i][j] for i in range(len(X))]
                    )
                    
                b -= lr * np.mean(
                    [(self.predict(X[i], w, b) - Y[i]) for i in range(len(X))]
                )
                """
                # victorized implementation
                temp_w -= lr * 1/m * (X@w + b - Y)@X
                b -= lr * np.mean(X@w + b - Y)
                w = temp_w.copy()
        
        # update the parameters of the model
        self.w = w
        self.b = b

    def cost(self, X=None, Y=None, Y_pred=None, w=None, b=None):
        if Y is None:
            raise TypeError("Y must be provided")
        
        if X is None and Y_pred is None:
            raise TypeError("X or Y_pred must be provided")
            
        if w is None:
            w = self.w

        if b is None:
            b = self.b
        
        if Y_pred is None:
            Y_pred = self.predict(X, w, b)
        
        m = len(Y)
        return np.mean((Y_pred - Y) ** 2)
        
    def predict(self, X, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        return X@w + b


if __name__ == "__main__":

    def f(x):
        w = np.array([-10, 10, -2])
        return np.dot(w, x.T) + 2 # b = 2
    
    # create training data
    X_train = np.random.random((1000, 3))
    Y_train = f(X_train)
    
    # initiate and train the model
    model = LineaRegression()
    model.train(X_train, Y_train, learning_curve=True)
    
    # create test data
    X_test = np.random.random((1000, 3))
    Y_test = f(X_test)
    # make prediction
    Y_pred = model.predict(X_test)

    print(f"MSE: {model.cost(Y=Y_test, Y_pred=Y_pred)}")
    print(f"Y_test mean value: {np.mean(Y_test)}")
    print(f"Y_pred mean value: {np.mean(Y_pred)}")
    print(f"Model w: {model.w}")
    print(f"Model b: {model.b}")