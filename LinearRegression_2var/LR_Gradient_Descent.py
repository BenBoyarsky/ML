import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
ml_folder = Path.cwd().parent
sys.path.insert(0, str(ml_folder))
from Data_Creator import linear_data_creator

def mean_square_error(x, y, w, b) -> float:
    m = len(x)
    return (1/2 * m) * ((w*x+b - y) ** 2).sum()

def mean_square_error_gradient(x, y, w, b) -> tuple[float, float]:
    m = len(x)
    dJ_dw = (1/m) * ((w*x+b - y) * x).sum()
    dJ_db = (1/m) * (w*x+b - y).sum()
    return (dJ_dw, dJ_db)

def gradient_descent(x, y, w_in = 0, b_in = 0, cost_function = mean_square_error, gradient_function = mean_square_error_gradient, alpha = 0.01, iterations = 5000) -> tuple[float, float]:
    w_history = [w_in]
    b_history = [b_in]
    cost_history = [cost_function(x, y, w_in, b_in)]
    min_cost_index = 0
    for i in range(iterations):
        gradient = gradient_function(x, y, w_history[i], b_history[i])
        w = w_history[i] - alpha * gradient[0]
        w_history.append(w)
        b = b_history[i] - alpha * gradient[1]
        b_history.append(b)
        cost = cost_function(x, y, w, b)
        cost_history.append(cost)
        if cost < cost_history[min_cost_index]:
            min_cost_index = i

    w_best = round(w_history[min_cost_index], 2)
    b_best = round(b_history[min_cost_index], 2)
    
    plt.scatter(x, y, color="blue", marker="x", label="Noisy Data")
    plt.axline((0, b_best), slope=w_best, color="red", label= f"Best Fit: y = {w_best}x + {b_best}")
    plt.title("Gradient Descent on Random Data")
    plt.legend()
    plt.show()
    return w_best, b_best

def main():
    print("=====Gradient Descent Verifier on Arbitrary Line=====")
    w = float(input("Weight: "))
    b = float(input("Bias: "))
    data = linear_data_creator(w, b)
    X = data[:,0].copy()
    Y = data[:,1].copy()
    w_best, b_best = gradient_descent(X, Y)

if __name__ == '__main__':
    main()