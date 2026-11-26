import numpy as np
from typing import Union

rng = np.random.default_rng()

#exclusive of upper bound
def random_num_in_range(range) -> float:
    return range[0] + (rng.random() * (range[1] - range[0]))

#tolerance is the percent as a decimal that data can differ from the original line based on
#the midpoint of the line
def linear_data_creator(w, b, x_range = [1, 10], tolerance = 0.05, num = 50) -> np.ndarray:
    num
    data = []
    noise = (w * ((x_range[1] + x_range[0]) / 2) + b) * tolerance
    for i in range(num):
        x = random_num_in_range(x_range)
        y = random_num_in_range([(w * x + b) - noise, (w * x + b) + noise])
        data.append([x,y])
    return np.vstack(data)

def multiple_linear_data_creator(W: np.ndarray, b: Union[float, int] , x_range: np.ndarray = None, tolerance = 0.05, num_samples = 50) -> np.ndarray:
    m = len(W)
    if x_range is None:
        data = []
        for i in range(m):
            data.append([1, 10])
        x_range = np.vstack(data)
    randomInputs = np.random.uniform(x_range[:, 0], x_range[:, 1], (num_samples, m))
    actualOutputs = randomInputs @ W + b
    noisyOutputs = actualOutputs + np.random.normal(0, tolerance, actualOutputs.shape)
    noisyOutputsAsColumnVector = np.vstack(noisyOutputs)
    return randomInputs, noisyOutputsAsColumnVector
