from .base_fusion import BaseFusion
import numpy as np


class NaiveFusion(BaseFusion):
    def __init__(self, eps=10**-8, act=False):
        super().__init__(eps=eps, act=act)


    def get_mapping(self, x: np.ndarray, y: np.ndarray):
        x = x[0]
        y = y[0]
        identity = np.eye(x.shape[0])
        return np.ones(x.shape[0]), np.zeros(x.shape[0]), np.ones(y.shape[0]), np.zeros(x.shape[0]), identity, identity