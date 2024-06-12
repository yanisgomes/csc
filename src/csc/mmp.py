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
        self.atom_pos = atom_pos
        self.atom_idx = atom_idx
        self.atom = dictionary.getAtom(atom_idx)
        self.atom_signal = self.atom.getAtomInSignal(signal_length=len(residual), offset=self.atom_pos)
        self.atom_info = {'x':self.atom_pos, 'b':self.atom.b, 'y':self.atom.y, 'sigma':self.atom.sigma}
        
        # Initialize the residual signal
        self.residual = residual
        self.mse = np.linalg.norm(self.residual)**2

        # Initialize the parent node for backtracking
        self.parent = parent
        self.children = []

    def get_depth(self) -> int:
        if self.parent is  None :
            return 0
        else :
            return self.parent.get_depth() + 1

    def add_child(self, node_idx:int, atom_pos:int, atom_idx:int):
        """
        Add a child node to the current node with an additional atom in the approximation.
        """
        new_residual = self.residual - self.atom_signal
        child_node = MMPNode(node_idx, self.dictionary, atom_pos, atom_idx, new_residual, parent=self)
        self.children.append(child_node)
        return child_node

class MMPTree() :

    def __init__(self, dictionary:ZSDictionary, signal:np.ndarray, sparsity:int, connections:int) :
        self.dictionary = dictionary
        self.signal = signal
        self.sparsity = sparsity
        self.connections = connections
        self.root = None