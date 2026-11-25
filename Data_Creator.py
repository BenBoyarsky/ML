import numpy as np

rng = np.random.default_rng()

#exclusive of upper bound
def random_num_in_range(range) -> float:
    return range[0] + (rng.random() * (range[1] - range[0]))

#tolerance is the percent as a decimal that data can differ from the original line based on
#the midpoint of the line
def linear_data_creator(w, b, x_range = [1, 10], tolerance = 0.05, num = 50) -> np.ndarray:
    data = []
    noise = (w * ((x_range[1] + x_range[0]) / 2) + b) * tolerance
    for i in range(num):
        x = random_num_in_range(x_range)
        y = random_num_in_range([(w * x + b) - noise, (w * x + b) + noise])
        data.append([x,y])
    return np.vstack(data)

def multiple_linear_data_creator(W: np.ndarray, b: float | int, x_range: np.ndarray = None, tolerance = 0.05, num = 50) -> np.ndarray:
   
    #coolest methods of all time begin
    def GenerateRandomInputs(num: int, x_range: np.ndarray) -> np.ndarray:
        randomInputsAsList = []
        for i in range(num):
            inputs = []
            for inputRange in x_range:
                inputInRange = random_num_in_range([inputRange[0], inputRange[1]])
                inputs.append(inputInRange)
            randomInputsAsList.append(inputs)
        randomInputs = np.vstack(randomInputsAsList)

        return randomInputs

    def GenerateActualOutputs(weights: np.ndarray, bias: float, randomInputs: np.ndarray):
        actualOutputsAsList = []

        for randomInput in randomInputs:
            actualOutput = np.dot(weights, randomInput) + bias
            actualOutputsAsList.append(actualOutput)
        return np.vstack(actualOutputsAsList)


    def FindNoises(tolerance: float, weights: np.ndarray, x_range: np.ndarray, bias: float) -> np.ndarray:
        noisesAsList = []

        i = 0
        for weight in weights:
            inputRange = (x_range[i][0], x_range[i][1])
            initialPoint = inputRange[0]*weight + bias
            finalPoint = inputRange[1]*weight + bias
            midpoint = (initialPoint + finalPoint)/2
            noise = midpoint*tolerance

            noisesAsList.append(noise)
            i += 1

        noises = np.vstack(noisesAsList)
        return noises

    def FindNoisyOutputs(actualOutputs: np.ndarray, noises: np.ndarray):
        noisyOutputsAsList = []
        i = 0
        for actualOutput in actualOutputs:
            noise = noises[i]
            noiseRange = (actualOutputs[i] - noise, actualOutputs[i] + noise)
            noisyOutput = random_num_in_range(noiseRange)
            noisyOutputsAsList.append(noisyOutput)
            i += 1
        return np.vstack(noisyOutputsAsList)
    #coolest methods of all time end

    m = len(W)
    if x_range is None:
        data = []
        for i in range(m):
            data.append([1, 10])
        x_range = np.vstack(data)

    randomInputs = GenerateRandomInputs(num, x_range)
    actualOutputs = GenerateActualOutputs(W, b, randomInputs)
    noises = FindNoises(tolerance, W, x_range, b)
    noisyOutputs = FindNoisyOutputs(actualOutputs, noises)

    pointsAsList = []
    i = 0
    for noisyOutput in noisyOutputs:
        point = (randomInputs[i], noisyOutput)
        pointsAsList.append(point)
        i += 1
    return np.vstack(pointsAsList)

