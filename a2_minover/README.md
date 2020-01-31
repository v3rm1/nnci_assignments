## Assignment 2: Learning a Rule

### Runtime Instructions
1. The program is written in python v 3.x and depends on the numpy package.
2. Install the numpy, pandas and matplotlib packages using the followig pip command:

    ```pip install numpy pandas matplotlib```
3. The program is contained in the ```perceptron_minover.py``` file. Run the file to simulate the results of training a perceptron using the MinOver and AdaTron algorithms on randomized datasets.

The data is randomized for different $\alpha$ and $N$ values, preset to the following ranges:
* $\alpha$ = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
* $N$ = 4

50 iterations of the Adatron and MinOver algorithms are simulated and thr resulting outputs are stored as csv files and graphs in the output directory.
