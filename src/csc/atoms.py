import numpy as np
from typing import Tuple, List
from itertools import product
import matplotlib.pyplot as plt

def der_zs_formula(b, sigma, x, y):
    """Derivatives (with respect to x) of Zatsepin and Shcherbinin formulas.

    Args:
        b (float): (2 * b) is the width of the defect.
        s = sigma (float): Linear density of the magnetic field.
        x (array_like): Vector of longitudinal position (the defect is at x=0).
        y (float): Lateral position (=distance to the surface).

    Returns:
        der_Hr (array_like): Vector dHr/dx (derivative of the magnetic field
            radial component). Same size as x.
        der_Ha (array_like): Vector dHa/dx (derivative of the magnetic field
            axial component). Same size as x.
    """
    denom = ((x + b) ** 2 + y**2) * ((x - b) ** 2 + y**2)
    der_denom = 2 * (x + b) * ((x - b) ** 2 + y**2) + ((x + b) ** 2 + y**2) * 2 * (
        x - b
    )
    num = -8 * sigma * b * x * y
    der_num = -8 * sigma * b * y
    der_Hr = (der_num * denom - num * der_denom) / denom**2

    num = -4 * sigma * b * (x**2 - y**2 - b**2)
    der_num = -4 * sigma * b * 2 * x
    der_Ha = (der_num * denom - num * der_denom) / denom**2

    return der_Hr, der_Ha

class ZSAtom() :
    STEP = 0.001
    SUPPORT_THRESHOLD = 0.001

    def __init__(self, b, y, sigma) :
        # Atom step
        self.step = ZSAtom.STEP
        # Support threshold
        self.support_threshold = ZSAtom.SUPPORT_THRESHOLD
        # Apply the ZS formula
        xmin = -b - 1
        xmax = b + 1
        self.x = np.linspace(xmin, xmax, int((xmax - xmin)/self.step))
        der_hr, _ = der_zs_formula(b, sigma, self.x, y)

        # self.values is the normalized signal of the atom
        self.values = der_hr / np.linalg.norm(der_hr)
        new_sigma = sigma / np.linalg.norm(der_hr)

        # Save the parameters
        self.b = b
        self.y = y
        self.sigma = new_sigma
        self.params = {'b': b, 'y': y, 'sigma': new_sigma}

        # Find the largest interval where the template is not entirely smaller than a threshold.
        tmp = np.diff((self.values > ZSAtom.SUPPORT_THRESHOLD).astype(int))
        start = np.argwhere(tmp == 1).flatten()[0]
        end = np.argwhere(tmp == -1).flatten()[-1]
        self.x = self.x[start:end]
        self.values = self.values[start:end]
        # Compute the support width
        self.support_width = len(self.values)
        self.var = np.var(self.values)
        # Compute the position starting from 0
        self.position = self.x - self.x[0]

    @classmethod
    def from_params(cls, params) -> 'ZSAtom':
        return cls(params['b'], params['y'], params['sigma'])

    def __str__(self) -> str:
        return "b={:.2f}, y={:.2f}, sigma={:.2e}".format(self.params['b'], self.params['y'], self.params['sigma'])

    def __getitem__(self, key):
        return self.params.get(key, None)
    
    def __setitem__(self, key, value):
        self.params[key] = value

    def __len__(self) -> int:
        return len(self.values)
    
    def __call__(self, *args, **kwargs):
        return self.getSignal(*args, **kwargs)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            self.sigma /= other
            self.params['sigma'] = self.sigma
            self.values /= other
            return self
        else:
            raise TypeError("Unsupported operand type for /: 'ZSDictionary' and '{}'".format(type(other).__name__))
    
    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            self.atoms = [atom / other for atom in self.atoms]
            return self
        else:
            raise TypeError("Unsupported operand type for /=: 'ZSDictionary' and '{}'".format(type(other).__name__))
    
    def copy(self) -> 'ZSAtom':
        return ZSAtom(self.params['b'], self.params['y'], self.params['sigma'])

    def set_param(self, key, value):
        self.params[key] = value
    
    def get_param(self, key):
        return self.params[key]
    
    def getInfos(self) -> dict:
        return self.params

    def getPlots(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.position , self.values
    
    def padBothSides(self, signal_length) :
        """Pad the values to the desired length"""
        assert signal_length >= len(self), "signal_length must be greater than or equal to self.support_width"
        # Compute the left and rigth padding
        total_padding = signal_length - len(self)
        padding_left = total_padding // 2
        padding_right = total_padding - padding_left
        # Pad the values
        self.values = np.pad(self.values, (padding_left, padding_right), mode='constant', constant_values=0)
        # Pad the x values
        self.x_pad_left = self.x[0] - self.step * np.arange(padding_left, 0, -1)
        self.x_pad_right = self.x[-1] + self.step * np.arange(1, padding_right + 1)
        self.x = np.concatenate([self.x_pad_left, self.x, self.x_pad_right])
        self.position = self.x - self.x[0]

    def getNoiseVarToSNR(self, snr) -> float:
        return self.var / (10 ** (snr / 10))

    def getSignal(self, *args, **kwargs) -> np.ndarray:
        """Returns the ZS atom signal
        Args:
            signal_length (int): The desired length of the atom signal
            offset (int): The offset of the atom signal
        Returns:
            np.ndarray: The ZS atom signal
        """
        return self.values