import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt

class Perceptron:
    def __init__(self, epochs=100, c = 0):
        self.epochs = epochs
        self.c = c

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
            n_correct = 0
            for s_idx, sample in enumerate(X):
                pred_corr = self.predict_correct(sample, wt, y[s_idx])
                n_correct += pred_corr
                if not pred_corr:
                    del_wt = (1/X.shape[1])*y[s_idx]*sample
                    wt = np.add(wt, del_wt)
                    hist_emb_strength[s_idx] += 1
            acc.append(n_correct / X.shape[0])
            if acc[-1] == 1:
                return 1
        return 0
    
    def predict_correct(self, x, wt, y):
        """
        Generate a prediction output for a given datapoint based on the application of the activation function on the datapoint and the weight vector.
        Arguments:
            x: Input data point of the shape [1*m | m = number of features]
            wt: weight vector of the shape [1*m | m = number of features] from the trained perceptron
        Returns:
            Prediction label that is either +1 or -1
        """
        E_mu = np.dot(x, np.transpose(wt))*y
        if E_mu > c:
            return 1
        else:
            return 0

if __name__ == "__main__":

    N_range = [10, 20]
    c_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    
    plot_data = pd.DataFrame(columns=["N", "P", "Separable_Count", "alpha"])
    for n in N_range:
        print("N: ", n)
        for c in c_range:
            p = n*2
            model = Perceptron(200,c)
            print("c: ", c)
            seperable_count = 0
            for i in range(0,500):
                X, y = model.generate_data(n, p)
                if model.fit(X,y):
                    seperable_count += 1
            plot_data = plot_data.append({"N": n, "c": c, "Separable_Count": seperable_count/200}, ignore_index=True)
    plot_data.to_csv("Plot_Data_c.csv")
    
    x = c_range
    y1 = plot_data[plot_data["N"]==10]["Separable_Count"]
    y2 = plot_data[plot_data["N"]==20]["Separable_Count"]
    fig, ax = plt.subplots()
    ax.plot(x,y1,c='b',marker="^",ls='--',label='N = 10',fillstyle='none')
    ax.plot(x,y2,c='g',marker=(8,2,0),ls='--',label='N = 20')

    ax.set(xlabel='c', ylabel='Probability of l.s.', title='prob l.s / c')
    ax.grid()
    plt.legend(loc=1)

    fig.savefig("c vs Probability of Linear Separability.png")
    plt.show()