import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt

class Perceptron:
    def __init__(self, epochs=100):
        self.epochs = epochs

    def generate_data(self, n_features, p_data):
        X = np.zeros((p_data, n_features))
        y = np.zeros(p_data)
        for i in range(p_data):
            for j in range(n_features):
                X[i,j] = random.gauss(0,1)
            y[i] = (random.randint(0,1)*2)-1

        return X, y
    
    def fit(self, X, y):
        """
        Train the perceptron to map input vectors to output vectors.
        Learning rate affects the modification of the weight vector in each training step.
        Arguments:
            X: Input features, type: numpy array of shape [n*m | n = number of samples, m = number of features]
            y: Output vector, type: numpy array of shape [n | n = number of samples]
            
        Returns:
            None
        """
        # Defining a weight vector, randomly initialised to values between 0 and 1. shape: 
        wt = np.zeros(X.shape[1])

        # Defining the training strategy
        acc = []
        hist_emb_strength = np.zeros(y.shape)
        for epoch in range(self.epochs):
            y_hat = []
            for s_idx, sample in enumerate(X):
                pred = self.predict(sample, wt)
                y_hat.append(pred)
                if not pred == y[s_idx]:
                    del_wt = y[s_idx]*sample
                    wt = np.add(wt, del_wt)
                    hist_emb_strength[s_idx] += 1
            acc.append(self._compute_acc(y, y_hat))
            if acc[-1] == 1:
                return 1
        return 0

    def _compute_acc(self, y, y_hat):
        """
        Compute the accuracy of the perceptron. Accuracy = number of correctly classified datapoints / total number of data points
        Arguments:
            y: Actual label for a given datapoint, is either +1 or -1
            y_hat: Label predicted by the perceptron, is either +1 or -1
        Returns:
            Accuracy as floating point value between 0 and 1
        """
        return (sum(np.array(y_hat) == np.array(y)) / float(len(y)))
    
    def predict(self, x, wt):
        """
        Generate a prediction output for a given datapoint based on the application of the activation function on the datapoint and the weight vector.
        Arguments:
            x: Input data point of the shape [1*m | m = number of features]
            wt: weight vector of the shape [1*m | m = number of features] from the trained perceptron
        Returns:
            Prediction label that is either +1 or -1
        """
        sum = np.sum(np.dot(x, np.transpose(wt)))
        if sum > 0.0:
            return 1
        else:
            return -1

if __name__ == "__main__":
    N_range = [8, 20, 40]
    A_range = [0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]

    model = Perceptron(500)
    plot_data = pd.DataFrame(columns=["N", "P", "Separable_Count", "alpha"])
    for n in N_range:
        print("N: ", n)
        for a in A_range:
            p = int(a*n)
            print("P: ", p)
            seperable_count = 0
            for i in range(0,50):
                X, y = model.generate_data(n, p)
                if model.fit(X,y):
                    seperable_count += 1
            plot_data = plot_data.append({"N": n, "P": p, "Separable_Count": seperable_count/50, "alpha": a}, ignore_index=True)
    plot_data.to_csv("Plot_Data.csv")
    
    x = A_range
    y1 = plot_data[plot_data["N"]==8]["Separable_Count"]
    y2 = plot_data[plot_data["N"]==20]["Separable_Count"]
    y3 = plot_data[plot_data["N"]==40]["Separable_Count"]
    fig, ax = plt.subplots()
    ax.plot(x,y1,c='b',marker="^",ls='--',label='N = 8',fillstyle='none')
    ax.plot(x,y2,c='g',marker=(8,2,0),ls='--',label='N = 20')
    ax.plot(x,y3,c='k',ls='-',label='N = 40')

    ax.set(xlabel='alpha', ylabel='Probability of l.s.', title='Prob/alpha')
    ax.grid()
    plt.legend(loc=1)

    fig.savefig("alpha vs Probability of Linear Separability.png")
    # plt.show()