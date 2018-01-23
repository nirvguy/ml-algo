import numpy as np

def forgy_initialization(data, k):
    return [data[index] for index in np.random.choice(range(len(data)), k)]
