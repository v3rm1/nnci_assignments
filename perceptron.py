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

    #TODO: Define _compute_acc
    #TODO: Define predict
    #TODO: Define wt*sample function