"""
This is univariant implementation of logistic regression that i made for learning purposes
it can be usefull for Simplification.

this implementation also has plot_cost function that is not availabe in the full implementation it worth trying :)
"""
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, w=0, b=0, lr=1):
        self.w = w
        self.b = b
        self.lr = lr

    def train(self, X, Y, n_iters=300, learning_curve=False, lr=0):
        w = self.w
        b = self.b
        if lr == 0:
            lr = self.lr
        
        temp_w = w # will be used for simultaneous update
        m = len(X)
        
        # I could check `if learning_curve` inside the training loop but i made it like this so the check happen one time instead of n_iters times
        if learning_curve:
            costs = []
            for _ in range(n_iters):
                # non victorized implementation try it and compare the time complexity
                """
                temp_w = w - lr * np.mean(
                    [(self.predict(X[i], w, b) - Y[i]) * X[i] for i in range(len(X))]
                )
                b = b - lr * np.mean(
                    [(self.predict(X[i], w, b) - Y[i]) for i in range(len(X))]
                )
                """
                # victorized implementation
                temp_w -= lr * 1/m * (self.predict(X, w, b) - Y)@X
                b -= lr * np.mean(self.predict(X, w, b) - Y)
                w = temp_w
                costs.append(self.cost(X=X, Y=Y, w=w, b=b))
                
            # print(costs[::50]) # for debugging purposes    
            plt.plot(list(range(1, n_iters + 1)), costs)
            plt.title("change in cost funtion value throug iterations")
            plt.xlabel("iteration")
            plt.ylabel("cost function value")
            
            if not os.path.exists("plots"):
               os.mkdir("plots")
            # save the figure on ./plots/uni_logistic_learning_curve.png
            plt.savefig("plots/uni_logistic_learning_curve.png", dpi=300)
        
        else:    
            for _ in range(n_iters):
                # temp_w = w - lr * np.mean(
                #     [(self.predict(X[i], w, b) - Y[i]) * X[i] for i in range(len(X))]
                # )
                # b = b - lr * np.mean(
                #     [(self.predict(X[i], w, b) - Y[i]) for i in range(len(X))]
                # )
                
                temp_w -= lr * 1/m * (self.predict(X, w, b) - Y)@X
                b -= lr * np.mean(self.predict(X, w, b) - Y)
                w = temp_w
                
        # update the parameters of the model
        self.w = w
        self.b = b

    def predict(self, X, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b

        return self.sigmoid(w * X + b)
    
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
        
        # non victorized implementation
        """
        return - np.mean([
            (1 - Y[i]) * np.log(1 - Y_pred[i])    
            + Y[i] * np.log(Y_pred[i])
            for i in range(len(Y_pred))
        ])
        """
        # victorized implementation
        m = len(Y)            
        return -1/m * ((1 - Y).T@np.log(1 - Y_pred) + Y@np.log(Y_pred))
        
    
    def score(self, y_test, y_pred):
        # round both of they can be intered in any order
        y_test = np.round(y_test)
        y_pred = np.round(y_pred)
        
        equals = (y_test == y_pred) # the result is an array of boolean values that result from evaluating the equality of each two corresponding elements
        return equals.mean() # the `False` values treated like 0s and `True` values treated like 1s so the result is a number between 0 and 1 represent the ratio of `True` values in the array (i.e. the ratio of correct predictions that the model made)
        
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
      
    def plot_cost(self, X, Y):
        # initiate experimental weights
        ws = np.linspace(-5, 5, 2000)
        bs = np.linspace(-5, 5, 2000)
        # calculate the cost for the experimental wieghts
        costs = [self.cost(X=X, Y=Y, w=ws[i], b=bs[i]) for i in range(len(ws))]
        
        # print(costs[::100]) # for debugging purposes
        
        # prepare the axises matrices
        X, Y = np.meshgrid(ws, bs)
        Z, _ = np.meshgrid(costs, bs)
        
        # make the 3d figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z)
        
        ax.set_xlabel('w')
        ax.set_ylabel('b')
        ax.set_zlabel('J(w, b)')
        ax.set_title('3D Plot of logistic regression cost function')
        ax.view_init(elev=30, azim=45)  # Adjust the viewing angle
        
        if not os.path.exists("plots"):
           os.mkdir("plots") 
        # save the figure into .plots/logistic-cost-3d.png
        plt.savefig("plots/logistic-cost-3d.png", dpi=300)
        plt.show()
    

if __name__ == "__main__": # so the follwing code will not be exicuted when the model is imported
    X = np.linspace(-3, 3, 1000) # create the data
    i = X >= 0 # the result is an array of Booleans, with `True` correspond to the elements those achieve the condition and `False` to those don't
    Y = np.zeros(1000) # initiate the target with 0s
    Y[i] = 1. # the elements correspond to elements of X those achive the codition updated to 1.
    
    x_train = X[:800]
    y_train = Y[:800]
    x_test = X[800:]
    y_test = Y[800:]
    
    model = LogisticRegression()
    t1 = time()
    model.train(x_train, y_train, learning_curve=True)
    y_pred = model.predict(x_test)
    t2 = time()
    
    print(f"Cost: {model.cost(Y=y_test, Y_pred=y_pred)}")
    print(f"Score: {model.score(y_test, y_pred)}")
    print(f"Time: {t2 - t1}s")
    print(f"W: {model.w}s")
    print(f"B: {model.b}s")
    model.plot_cost(X, Y)