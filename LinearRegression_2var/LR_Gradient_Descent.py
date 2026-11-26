import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
print(sys.version)
from pathlib import Path
ml_folder = Path(__file__).resolve().parent.parent
print(ml_folder)
sys.path.insert(0, str(ml_folder))
from Data_Creator import linear_data_creator
from Data_Creator import multiple_linear_data_creator

    #Y & W are column vectors
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

def gradient_descent(X: np.ndarray, Y: np.ndarray, W_in = None, b_in = 0, cost_function = mean_square_error, gradient_function = mean_square_error_gradient, alpha = 0.01, iterations = 50000):
    num_features = X.shape[1]
    
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



def ParseList(listAsString: str) -> np.ndarray:
    if(listAsString == ""):
        print("you're a moron")
        return np.ndarray(42)

    listAsString = listAsString.strip().removeprefix("[").removesuffix("]")
    listAsListOfStrings = listAsString.split(",")
    for i in range(len(listAsListOfStrings)):
        listAsListOfStrings[i] = listAsListOfStrings[i].strip()
    listAsListofStringsConvertedToFloats = []
    for element in listAsListOfStrings:
        value = ParseForFloat(element)
        if value is not None:
            listAsListofStringsConvertedToFloats.append(value)
    return np.array(listAsListofStringsConvertedToFloats)


def ParseForFloat(string: str) -> float:
    try:
        value = float(string)
        pass
    except:
        value = None
        pass
    return value


def main():
    print("=====Gradient Descent Verifier on Arbitrary Line=====")
    print("Weights: '[w₁, w₂, w₃, ...]'")
    w = input()
    W = ParseList(w)
    b = float(input("Bias: "))
    data = np.ndarray
    if(len(W) == 1):
        weight = W[0]
        data = linear_data_creator(weight, b)
        X = data[:,0].copy()
        Y = data[:,1].copy()
    else:
        X, Y = multiple_linear_data_creator(W, b)
    w_best, b_best = gradient_descent(X, Y)
    print(w_best)
    print(b_best)

if __name__ == '__main__':
    main()
