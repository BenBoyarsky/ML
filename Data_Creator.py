import numpy as np

rng = np.random.default_rng()

#exclusive of upper bound
def random_num_in_range(range) -> float:
    return range[0] + (rng.random() * range[1])

def linear_data_creator(w, b, x_range = [0, 10], tolerance = 0.05, num = 50) -> np.ndarray:
    data = []
    for i in range(num):
        x = random_num_in_range(x_range)
        y = random_num_in_range([(1 - tolerance) * (w * x + b), (1 + tolerance) * (w * x + b)])
        data.append([x,y])
    return np.vstack(data)





