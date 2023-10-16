"""
This is implementation of Univariant Linear Regression model with gradient decent algorithm and mean squared error as a cost function
"""
import os
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, w=0, b=0, lr=1):
        self.w = w
        self.b = b
        self.lr = lr

    def train(self, X, Y, n_iters=300, learning_curve=False, lr=0):
        w = self.w
        b = self.b
        if lr == 0:
            lr = self.lr
        
        temp_w = w # for simultaneous update
        m = len(X)
        
        if learning_curve: # I could check `if learning_curve` inside the training loop but i made it like this so the check happen one time instead of n_iters times
        
            costs = [] # for learning curve
            for _ in range(n_iters):
                # non visctorized implementaion
                """
                temp_w = w - self.lr * np.mean(
                    [(self.predict(X[i], w, b) - Y[i]) * X[i] for i in range(len(X))]
                )
                b = b - self.lr * np.mean(
                    [(self.predict(X[i], w, b) - Y[i]) for i in range(len(X))]
                )
                """
                # victorized implementation
                temp_w -= self.lr * 1/m * (self.predict(X, w, b) - Y)@X
                b -= self.lr * np.mean(self.predict(X, w, b) - Y)
                w = temp_w
                
                costs.append(self.cost(X=X, Y=Y, w=w, b=b))
            
            # print(costs[:10]) # for debugging purposes
            plt.plot(list(range(1, n_iters + 1)), costs)
            plt.title("change in cost funtion value throug iterations")
            plt.xlabel("iteration")
            plt.ylabel("cost function value")
            
            if not os.path.exists("plots"):
               os.mkdir("plots")
            # save the figure in ./plots/uni_linear_learning_curve.png
            plt.savefig("plots/uni_linear_learning_curve.png", dpi=300)
        
        else:    
            for _ in range(n_iters):
                # non vectorized implementaion
                """
                temp_w = w - self.lr * np.mean(
                    [(self.predict(X[i], w, b) - Y[i]) * X[i] for i in range(len(X))]
                )
                b = b - self.lr * np.mean(
                    [(self.predict(X[i], w, b) - Y[i]) for i in range(len(X))]
                )
                """
                
                # victorized implementation
                temp_w -= self.lr * 1/m * (self.predict(X, w, b) - Y)@X
                b -= self.lr * np.mean(self.predict(X, w, b) - Y)
                w = temp_w
        
        # update the mode parameters
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
            
        return w*X + b

    def plot_cost(self, X, Y):
        # create experimental weights
        ws = np.linspace(-100, 100, 2000)
        bs = np.linspace(-100, 100, 2000)
        costs = np.array([self.cost(X=X, Y=Y, w=ws[i], b=bs[i]) for i in range(len(ws))])
        print(costs[:10])
        X, Y= np.meshgrid(ws, bs)
        Z, _ = np.meshgrid(costs, costs)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z)
        
        ax.set_xlabel('w')
        ax.set_ylabel('b')
        ax.set_zlabel('J(w, b)')
        ax.set_title('3D Plot of Univariant Linear Regression model cost function (mse)')
        ax.view_init(elev=30, azim=45)  # Adjust the viewing angle

        if not os.path.exists("plots"):
           os.mkdir("plots")
        
        plt.savefig("./plots/lin_reg_3d_cost.png", dpi=300)
        plt.show()    
    
if __name__ == "__main__":
    def f(x):
        return 3 * x + 7 # w = 3 & b = 7
    
    # create the train data
    X_train = np.random.randn(100)
    Y_train = f(X_train)
    
    # initiate the model and train it using the trainign data
    model = LinearRegression()
    model.train(X_train, Y_train, learning_curve=True)
    
    # create the test data    
    X_test = np.random.randn(100)
    Y_test = f(X_test)
    Y_pred = model.predict(X_test)
    
    print(f"MSE: {model.cost(Y=Y_test, Y_pred=Y_pred)}")
    print(f"Y-test mean: {np.mean(Y_test)}")
    print(f"Y-pred mean: {np.mean(Y_pred)}")
    print(f"Model w: {model.w}")
    print(f"Model b: {model.b}")
    model.plot_cost(X_test, Y_test)
