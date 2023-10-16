import os
from time import time
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, w=np.zeros(1), b=0, lr=.3):
        self.w = w
        self.b = b
        self.lr = lr
        
    def train(self, X, Y, n_iters=2000, learning_curve=False, lr=0):
        w = self.w if self.w.shape == X.shape[1] else np.zeros(X.shape[1]) # initiate w with zeros if the provided w shape is not suitable
        b = self.b
        if lr == 0:
            lr = self.lr
        
        m = 1/len(X)
        temp_w = w.copy() # will be used for simultaneous update
        
        # I could check `if learning_curve` inside the training loop but i made it like this so the check happen one time instead of n_iters times 
        if learning_curve:
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
                temp_w -= lr * np.mean((self.predict(X, w, b) - Y)[:,np.newaxis] * X, axis=0)
                b -= lr * np.mean(self.predict(X, w, b) - Y)
                w = temp_w.copy()
                
                costs.append(self.cost(X=X, Y=Y, w=w, b=b))
            
            # print(costs[::50]) # for debugging purposes
            plt.plot(list(range(1, n_iters + 1)), costs)
            plt.title("change in cost funtion value throug iterations")
            plt.xlabel("iteration")
            plt.ylabel("cost function value")
            
            if not os.path.exists("plots"):
               os.mkdir("plots")
               
            # save the figure on ./plots/logistic_learning_curve.png
            plt.savefig("plots/logistic_learning_curve.png", dpi=300)
            plt.show()
        
        else:    
            for _ in range(n_iters):
                temp_w -= lr * np.mean((self.predict(X, w, b) - Y)[:,np.newaxis] * X, axis=0)
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
        
        return self.sigmoid(np.dot(w, X.T) + b)

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
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
        
        
        Y_pred = np.clip(Y_pred, 1e-10, 1-1e-10)
        
        # non victorized implementation
        """
        return - np.mean([
            (1 - Y[i]) * np.log(1 - Y_pred[i])    
            + Y[i] * np.log(Y_pred[i])
            for i in range(len(Y_pred))
        ])
        """
        # victorized implementation
        m = len(Y_pred)
        return - 1/m * ((1 - Y).T@np.log(1 - Y_pred) + Y.T@np.log(Y_pred))
        
    def score(self, y_test, y_pred):
        # round both of they can be intered in any order
        y_test = np.round(y_test)
        y_pred = np.round(y_pred)
        
        equals = y_test == y_pred # the result is an array of boolean values that result from evaluating the equality of each two corresponding elements
        
        return equals.mean() # the `False` values treated like 0s and `True` values treated like 1s so the result is a number between 0 and 1 represent the ratio of `True` values in the array (i.e. the ratio of correct predictions that the model made)
    
    def plot_decision_boundary(self, X, Y, boundary_x=None,boundary_y=None, **options):
        if X.shape[1] > 2:
            print("this function can just plot decision boundary for models that have 2 or less variables")
            return
            
        pos = X[Y == 1]
        neg = X[Y == 0]
        
        plt.scatter(pos.T[0], pos.T[1], marker=options["pos_marker"], c=options["pos_color"], s=options["scatter_size"])
        plt.scatter(neg.T[0], neg.T[1], marker=options["neg_marker"], c=options["neg_color"], s=options["scatter_size"])
        
        if boundary_x is None or boundary_y is None:
            x = np.linspace(-5, 5, 200)
            y = -(x * self.w[1] + self.b) / self.w[0]
        
        else:
            x = boundary_x
            y = boundary_y
        
        plt.plot(x, y, color=options["boundary_color"], linewidth=options["boundary_linewidth"])
        plt.title(options["figname"])
        plt.xlabel("w1")
        plt.xlabel("w2")
        
        if not os.path.exists("plots"):
           os.mkdir("plots")
        
        plt.savefig(f"plots/{options['figname']}.png", dpi=300)
        plt.show()
        
if __name__ == "__main__":# so the follwing code will not be exicuted when the model is imported
    np.random.seed(0)
    # create the data as a matrix of shape(1000, 2)
    
    # linear data give better performance
    # c = np.linspace(-5, 5, 1000)
    # X = np.column_stack((c1, c2))
    
    c1 = np.random.randn(1000)
    c2 = np.random.randn(1000)
    X = np.column_stack((c1, c2))
    
    w = np.array([3, 3]) # the vector w
    i = X@w + 2 > 0 # b = 2 # i is an array of booleans
    
    Y = np.zeros(1000) # initiate the target y as an array of zeros
    Y[i] = 1. # the elements of y correspond to element of X that achieve the condition replaced with 1.
    
    x_train = X[:800]
    y_train = Y[:800]
    x_test = X[800:]
    y_test = Y[800:]
    
    model = LogisticRegression()
    t1 = time() # for measuring time complexity
    model.train(x_train, y_train, learning_curve=True, lr=.111)
    y_pred = model.predict(x_test)
    t2 = time()
    
    print(f"Cost: {model.cost(Y=y_test, Y_pred=y_pred)}")
    print(f"Score: {model.score(y_test, y_pred)}")
    print(f"Time: {t2-t1}s") # the time of trainign + prediction
    
    # print the wieghts the model choosed
    print(f"W: {model.w}")
    print(f"B: {model.b}")
    model.plot_decision_boundary(X, Y, pos_color="blue", pos_marker="o", neg_color="red", neg_marker="x",scatter_size=20, boundary_color="green", boundary_linewidth=1, figname="desicsion_boundary")
    
    
    """
    # the next code fit a polynomial model (circular) on the data and plot the decision boundary
    # comment out the previous code and give it a try :)
    
    model2 = LogisticRegression()
    # generate the trainign data
    c1 = np.random.randn(1000)*3
    c2 = np.random.randn(1000)*3
    X_train = np.column_stack((c1, c2))
    
    # generate the circular function (outside the circle y=1, inside y=0)
    X_train_piped = X_train ** 2
    i = [x[0] + x[1] - 9 > 0 for x in X_train_piped]
    Y_train = np.zeros(1000)
    Y_train[i] = 1.
    
    # train the model and print choosen weights
    model2.train(X_train_piped, Y_train)
    print(model2.w)
    print(model2.b)
    
    # generate the decision boundary
    x = np.linspace(-9, 9, 1000)
    y = ((-model2.b - model2.w[0]*x**2)/model2.w[1])**.5
    x = np.column_stack((x, x))
    y = np.column_stack((y, -y))
    
    # plot the decision boundary
    model2.plot_decision_boundary(X_train, Y_train, boundary_x=x, boundary_y=y, pos_color="black", pos_marker="+", neg_color="yellow", neg_marker="o",scatter_size=20, boundary_color="green", boundary_linewidth=1, figname="logistic_circular_function_decision_boundary")
    """