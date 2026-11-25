import numpy as np

rng = np.random.default_rng()

#exclusive of upper bound
def random_num_in_range(range) -> float:
    return range[0] + (rng.random() * (range[1] - range[0]))

#tolerance is the percent as a decimal that data can differ from the original line based on
#the midpoint of the line
def linear_data_creator(w, b, x_range = [1, 10], tolerance = 0.05, num = 50) -> np.ndarray:
    data = []
    noise = (w * ((x_range[1] - x_range[0]) / 2) + b) * tolerance
    for i in range(num):
        x = random_num_in_range(x_range)
        y = random_num_in_range([(w * x + b) - noise, (w * x + b) + noise])
        data.append([x,y])
    return np.vstack(data)

def multiple_linear_data_creator(w: np.ndarray, b, x_range: np.ndarray = None, tolerance = 0.05, num = 50) -> np.ndarray:
    m = len(w)
    if x_range == None:
        data = [[1, 10]]
        for i in range(m):
            data.append([1, 10])
        x_range = np.vstack(data)
    pass





