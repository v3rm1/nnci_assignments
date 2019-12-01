import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
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
        wt = np.random.rand(X.shape[1],)

        # Defining the training strategy
        for epoch in range(self.epochs):
            y_hat = []
            for s_idx, sample in enumerate(X):
                pred = self.predict(sample, wt)
                y_hat.append(pred)
                for f_idx, feature in enumerate(wt):
                    del_wt = self.learning_rate * (y[s_idx] - pred)
                    del_wt = del_wt * sample[f_idx - 1]
                    wt[f_idx - 1] = wt[f_idx - 1] + del_wt
            print(f"Epoch: {epoch}\tAccuracy: {self._compute_acc(y, y_hat)}")
        return

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