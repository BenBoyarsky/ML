import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
from pathlib import Path
ml_folder = Path.cwd().parent
sys.path.insert(0, str(ml_folder))
from Data_Creator import linear_data_creator
    #Y, W are column vectors
    #X is matrix w/ columns as features, and rows as individual samples


def mean_square_error(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b):
    num_samples = X.shape[0]
    Y_pred = X @ W + b
    residual = Y_pred - Y
    return (1 / (2*num_samples)) * ((residual)**2).sum()

def mean_square_error_gradient(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b) -> np.ndarray:
    num_samples, num_features = X.shape
    grad = np.zeros((num_features + 1, 1))

    Y_pred = X @ W + b
    residual = Y_pred - Y   
    grad[:-1] = (X.transpose() @ residual) / num_samples
    dJ_db = residual.mean()
    grad[-1] = dJ_db
    return grad

def gradient_descent(X: np.ndarray, Y: np.ndarray, W_in = None, b_in = 0, cost_function = mean_square_error, gradient_function = mean_square_error_gradient, alpha = 0.01, iterations = 5000):
    num_samples, num_features = X.shape
    
    if W_in is None:
       W_in = np.zeros((num_features, 1))
    W_b = np.vstack((W_in, [[b_in]])).astype(float)

    W_history = [W_in]
    b_history = [b_in]
    cost_history = [cost_function(X, Y, W_in, b_in)]
    min_cost_index = 0
    for i in range(iterations):
        grad = gradient_function(X, Y, W_history[i], b_history[i])
        W_b -= alpha * grad
        W_history.append(W_b[:-1].copy())
        b_history.append(W_b[-1].copy())
        cost = cost_function(X, Y, W_b[:-1], W_b[-1])
        cost_history.append(cost)
        if cost < cost_history[min_cost_index]:
            min_cost_index = i + 1
    W_best = W_history[min_cost_index]
    b_best = b_history[min_cost_index]
    return W_best, b_best

def main():
    pass

if __name__ == "__main__":
    main()