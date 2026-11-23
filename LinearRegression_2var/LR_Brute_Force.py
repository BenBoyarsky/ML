import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def mean_squared_error(y_true, y_pred):
    m = len(y_true)
    return (1 / (2 * m)) * ((y_pred - y_true) ** 2).sum()

def predict(x_true, w, b):
    return np.array(w * x_true + b)

def cost_function_graph(cost_function, x_real, y_real):
    w_vals = []
    b_vals = []
    cost_vals = []
    for w in np.arange(-10, 10, 0.05):
        for b in np.arange(-10, 10, 0.05):
            predicted_values = predict(x_real, w, b)
            w_vals.append(w)
            b_vals.append(b)
            cost_vals.append(cost_function(y_real, predicted_values))

    plt.scatter(w_vals, b_vals, c=cost_vals)
    plt.xlabel("w")
    plt.ylabel("b")
    plt.title("Cost Vs. w & b")
    plt.show()
            

data = pd.read_csv("../iris_petal_regression.csv")
x = np.array(data["petal_length_cm"])
y = np.array(data["petal_width_cm"])

cost_function_graph(mean_squared_error, x, y)


#CSV file
#Linear Regression
#