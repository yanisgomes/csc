from tqdm import tqdm
from numba import njit, jit
import numpy as np
from einops import rearrange
from functools import partial
from itertools import product
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, lsqr
from collections import Counter

from joblib import Parallel, delayed

from .dictionary import ZSDictionary
from .atoms import ZSAtom
from .utils import *

class MMPNode:
    def __init__(self, dictionary:ZSDictionary, atom_pos:int, atom_idx:int, residual:np.ndarray, parent=None):
        """
        Initialize an MMP node.

        Parameters:
        indices (list of int): List of indices of the selected atoms in the dictionary.
        dictionary (Dictionary): The dictionary containing all possible atoms.
        residual (np.array): The current residual signal.
        parent_approximation (np.array): The approximation of the signal by the parent node.
        """
        # Initialize the dictionary and the atom
        self.dictionary = dictionary
        self, atom_pos = atom_pos
        self.atom_idx = atom_idx
        self.atom = dictionary.getAtom(atom_idx)
        self.atom_signal = self.atom.getAtomInSignal(signal_length=len(residual))
        
        self.residual = residual
        self.parent = parent
        self.children = []

    def compute_approximation(self):
        """
        Compute the signal approximation based on the current set of indices.
        """
        approx = np.copy(self.parent_approximation)
        for idx in self.indices:
            atom = self.dictionary.get_atom(idx)
            projection = np.dot(self.residual, atom) * atom
            approx += projection
            self.residual -= projection
        return approx

    def add_child(self, atom_idx):
        """
        Add a child node to the current node with an additional atom in the approximation.
        """
        new_indices = self.indices + [new_index]
        new_residual = self.signal
        child_node = MMPNode(atom_idx, self.dictionary, new_residual, self.current_approximation)
        self.children.append(child_node)
        return child_node