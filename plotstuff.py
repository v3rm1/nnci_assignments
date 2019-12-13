import matplotlib.pyplot as plt
import pandas as pd

A_range = [1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3]

plot_data = pd.read_csv("Plot_Data2.csv")
    
x = A_range
y1 = plot_data[plot_data["N"]==10]["Separable_Count"]
y2 = plot_data[plot_data["N"]==20]["Separable_Count"]
y3 = plot_data[plot_data["N"]==30]["Separable_Count"]
y4 = plot_data[plot_data["N"]==40]["Separable_Count"]
fig, ax = plt.subplots()
ax.plot(x,y1,c='b',marker="^",ls='--',label='N = 10',fillstyle='none')
ax.plot(x,y2,c='g',marker=(8,2,0),ls='--',label='N = 20')
ax.plot(x,y3,c='k',ls='-',label='N = 30')
ax.plot(x,y4,c='c',ls='-',label='N = 40')

ax.set(xlabel='alpha', ylabel='Probability of l.s.', title='Prob/alpha')
ax.grid()
plt.legend(loc=1)

fig.savefig("alpha vs Probability of Linear Separability.png")
plt.show()