import numpy as np


def shuffle_and_divide(x, y, split_ratio=0.7):
    order = np.random.permutation(range(len(y)))
    x = x[order]
    y = y[order]

    split = int(split_ratio*len(y))
    train_x = x[:split]
    train_y = y[:split]

    test_x = x[split:]
    test_y = y[split:]
    return train_x, train_y, test_x, test_y