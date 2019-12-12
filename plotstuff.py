import matplotlib.pyplot as plt

x = [0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]

y1 = [1.0, 1.0, 0.96, 0.82, 0.56, 0.4, 0.3, 0.14, 0.1, 0.1]
y2 = [1.0, 1.0, 1.0, 0.9, 0.66, 0.36, 0.12, 0.0, 0.04, 0.0]
y3 = [1.0, 1.0, 1.0, 0.94, 0.8, 0.18, 0.08, 0.0, 0.0, 0.0]

fig, ax = plt.subplots()

ax.plot(x,y1,c='b',marker="^",ls='--',label='N = 8',fillstyle='none')
ax.plot(x,y2,c='g',marker=(8,2,0),ls='--',label='N = 20')
ax.plot(x,y3,c='k',ls='-',label='N = 40')

ax.set(xlabel='alpha', ylabel='Probability of l.s.', title='Prob/alpha')
ax.grid()
plt.legend(loc=2)

fig.savefig("test.png")
plt.show()