## Assignment 1: Perceptron Training

### Runtime Instructions
1. The program is written in python v 3.x and depends on the numpy package.
2. Install the numpy, pandas and matplotlib packages using the followig pip command:

    ```pip install numpy pandas matplotlib```
3. The program is contained in the ```perceptron.py``` file. Run the file to simulate the results of training a perceptron using the Rosenblatt algorithm on randomized datasets.

The data is randomized for different $\alpha$, $c$ and $N$ values, preset to the following ranges:
* $\alpha$ = [1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3]
* $N$ = [10, 30, 50]
* $c$ = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  

500 iterations of the Rosenblatt algorithms are simulated and the resulting outputs are stored as csv files and graphs in the output directory.
