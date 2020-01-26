import numpy as np
import random
from math import acos, pi
import pandas as pd
from matplotlib import pyplot as plt
import os

class Perceptron:
    def __init__(self, epochs=500, c = 0):
        self.epochs = epochs
        self.c = c
        self.threshold = pi/1000
    
    def generate_teacher_vec(self, n_features):
        w_str = np.zeros(n_features)
        for i in range(n_features):
            w_str[i] = random.gauss(0,1)
        return w_str/np.linalg.norm(w_str)

    def generate_data(self, n_features, p_data, w_str):
        X = np.zeros((p_data, n_features))
        y = np.zeros(p_data)
        for i in range(p_data):
            for j in range(n_features):
                X[i,j] = random.gauss(0,1)
            y[i] = np.sign(np.dot(w_str, X[i]))
            # print('X:{}\ny:{}\nw_str:{}'.format(X[i], y[i], w_str))

        return X, y
    
    def get_kappa(self, weight, X, y):
        kappa = np.zeros(X.shape[0])
        for i in range(len(kappa)):
            kappa[i] = (y[i] * np.dot(weight, X[i]))/np.linalg.norm(weight)
        # print("Kappa: {}".format(kappa))
        return kappa

    def minover_fit(self, X, y):
        stop_fit = False
        w = np.zeros(X.shape[1])
        converge_step = 0
        while not stop_fit:
            kappa = self.get_kappa(w, X, y)
            kappa_min = np.argmin(kappa)
            w_new = w + (X[kappa_min] * y[kappa_min])/X.shape[1]
            # Return the minimum between normalized dot product and 1, since it is observed that the dot product is sometimes greater than 1, due to floating point calculation errors
            w_angle = acos(min(np.dot(w,w_new)/(np.linalg.norm(w)*np.linalg.norm(w_new)), 1))
            if np.abs(w_angle) < self.threshold:
                stop_fit = True
            w = w_new
            converge_step += 1
        return w_new, converge_step

    def adatron_fit(self, X, y):
        stop_fit = False
        emb_str = np.zeros(X.shape[0])
        w = np.zeros(X.shape[1])
        converge_step = 0
        while not stop_fit:
            w_old = w
            for i in range(X.shape[0]):
                E = np.dot(w, X[i]) * y[i]
                emb_str_update = max(0, emb_str[i] + (1 - E))
                w = w + ((emb_str_update - emb_str[i]) * X[i])/(X.shape[1])
                emb_str[i] = emb_str_update
            w_angle = acos(min(np.dot(w,w_old)/(np.linalg.norm(w)*np.linalg.norm(w_old)), 1))
            if np.abs(w_angle) < self.threshold:
                stop_fit = True
            converge_step += 1
        return w, converge_step


if __name__ == "__main__":
    file_path = os.getcwd()
    print(file_path)
    A_range = np.arange(0.1, 5, 0.1)
    N = 10
    

    # Dataframe to store the outputs of the perceptron
    plot_data = pd.DataFrame(columns=["Alpha", "Generalization_Error", "Min_Kappa"])
    adatron_minover_compare = pd.DataFrame(columns=["Alpha", "Step_Count_Minover", "Step_Count_Adatron"])

    model = Perceptron(500)

    # Simulating the perceptron for multiple values of N and alpha
    for a in A_range:
        p = np.round(int(N*a))
        print("p_val: {}".format(p))
        gen_err = np.zeros(100)
        k_min = np.zeros(100)
        for i in range(0,100):
            w_star = model.generate_teacher_vec(N)
            X, y = model.generate_data(N, p, w_star)
            w_fit_minover, conv_step_minover = model.minover_fit(X, y)
            w_fit_adatron, conv_step_adatron = model.adatron_fit(X, y)
            gen_err[i] = acos(min(np.dot(w_fit_minover,w_star)/(np.linalg.norm(w_fit_minover)*np.linalg.norm(w_star)), 1))/pi
            k_min[i] = np.argmin(model.get_kappa(w_fit_minover, X, y))
        adatron_minover_compare = adatron_minover_compare.append({"Alpha": np.round(a), "Step_Count_Minover": conv_step_minover, "Step_Count_Adatron": conv_step_adatron}, ignore_index=True)
        avg_gen_err = np.mean(gen_err)
        kap_min = np.mean(k_min)
        plot_data = plot_data.append({"Alpha": np.round(a), "Generalization_Error": avg_gen_err, "Min_Kappa": kap_min}, ignore_index=True)
    plot_data.to_csv(os.path.join(file_path, "output/Plot_Data_alpha.csv"))
    adatron_minover_compare.to_csv(os.path.join(file_path, "output/Ada_Min_compare.csv"))

       
    # Defining a plotting function for plotting alpha versus Probability of linear separability.
    plt.clf()
    x = A_range
    y1 = plot_data["Generalization_Error"]
    fig, ax = plt.subplots()
    ax.plot(x,y1,c='b',marker="^",ls='--',fillstyle='none')
    ax.set(xlabel='Alpha', ylabel='Generalization Error', title='gen error / alpha')
    ax.grid()
    plt.legend(loc=1)

    fig.savefig(os.path.join(file_path, "output/a_vs_GE.png"))
    # plt.show()
    
    # Defining a plotting function for plotting alpha versus Kappa
    plt.clf()
    x = A_range
    y2 = plot_data["Min_Kappa"]
    fig, ax = plt.subplots()
    ax.plot(x,y2,c='r',marker="^",ls='--',fillstyle='none')
    ax.set(xlabel='Alpha', ylabel=r'$\kappa(t_{max})$', title=r'$\kappa(t_{max})$ vs Alpha')
    ax.grid()
    plt.legend(loc=1)

    fig.savefig(os.path.join(file_path, "output/a_vs_min_kap.png"))
    # plt.show()

    # Adatron and MonOver comparison plot
    plt.clf()
    x = A_range
    y3 = adatron_minover_compare["Step_Count_Minover"]
    y4 = adatron_minover_compare["Step_Count_Adatron"]
    fig, ax = plt.subplots()
    ax.plot(x,y3,c='g',marker=(8,2,0),ls='--',label='MinOver')
    ax.plot(x,y4,c='c',marker=(8,2,0),ls=':',label='AdaTron')
    ax.set(xlabel='Alpha', ylabel='Number of Iterations', title=r'Number of Iterations vs $\alpha$')
    ax.grid()
    plt.legend(loc=1)

    fig.savefig(os.path.join(file_path, "output/a_vs_step_ada_min.png"))
    # plt.show()
