import numpy as np
import os
from os.path import dirname, abspath


IRIS_X = dirname(abspath(__file__))+'/../../data/iris/processed/x.npy'
IRIS_Y = dirname(abspath(__file__))+'/../../data/iris/processed/y.npy'

WINE_X = dirname(abspath(__file__))+'/../../data/wine/processed/x.npy'
WINE_Y = dirname(abspath(__file__))+'/../../data/wine/processed/y.npy'

GLASS_X = dirname(abspath(__file__))+'/../../data/glass/processed/x.npy'
GLASS_Y = dirname(abspath(__file__))+'/../../data/glass/processed/y.npy'

DIABETES_X = dirname(abspath(__file__))+'/../../data/diabetes/processed/x.npy'
DIABETES_Y = dirname(abspath(__file__))+'/../../data/diabetes/processed/y.npy'

PIMA_X = dirname(abspath(__file__))+'/../../data/pima_diabetes/processed/x.npy'
PIMA_Y = dirname(abspath(__file__))+'/../../data/pima_diabetes/processed/y.npy'


def _load_data(x_path, y_path):
    with open(x_path, 'rb') as f:
        x = np.load(f)
    with open(y_path, 'rb') as f:
        y = np.load(f)
    return x, y


def load_iris(x_path=IRIS_X, y_path=IRIS_Y):
    return _load_data(x_path, y_path)


def load_glass(x_path=GLASS_X, y_path=GLASS_Y):
    return _load_data(x_path, y_path)


def load_wine(x_path=WINE_X, y_path=WINE_Y):
    return _load_data(x_path, y_path)


def load_diabetes(x_path=DIABETES_X, y_path=DIABETES_Y):
    return _load_data(x_path, y_path)


def load_pima_diabetes(x_path=PIMA_X, y_path=PIMA_Y):
    return _load_data(x_path, y_path)
