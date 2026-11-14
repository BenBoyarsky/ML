import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def mean_squared_cost(y, f_wb):
    m = len(y)
    return (1 / (2 * m)) * ((f_wb - y) ** 2).sum()

def predict(x, w, b):
    return np.array(w * x + b)

def LinearRegression(prediction_function)

def main():
    data = pd.read_csv("iris_petal_regression.csv")
    x = np.array(data["petal_length_cm"])
    y = np.array(data["petal_width_cm"])
    w = 0.3
    b = 0
    f_wb = w*x + b
    cost = mean_squared_cost(y, f_wb)
    print(f"Cost: {cost}")

    plt.scatter(x, y, color = "red", marker = "x")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.plot(x, f_wb)
    plt.show()
    

main()