import uuid
from tqdm import tqdm
from numba import njit, jit
import numpy as np
import networkx as nx
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
    def __init__(self, dictionary:ZSDictionary, signal:np.ndarray, activation_idx:int=None, parent=None):
        """
        Initialize an MMP node.

        Parameters:
        indices (list of int): List of indices of the selected atoms in the dictionary.
        dictionary (Dictionary): The dictionary containing all possible atoms.
        residual (np.array): The current residual signal.
        parent_approximation (np.array): The approximation of the signal by the parent node.
        """
        # Initialize the dictionary and its parameters
        self.dictionary = dictionary
        self.signal = signal
        self.nb_atoms = len(dictionary.getAtoms())
        self.atoms_length = dictionary.getAtomsLength()
        self.signal_length = len(signal)
        nb_valid_offset = self.signal_length - self.atoms_length + 1
        nb_activations = nb_valid_offset * self.nb_atoms

        # Initialize the parent node for backtracking
        self.parent = parent
        self.children = []

        # Initialize the activation mask
        if self.parent is None :
            self.activation_mask = np.zeros((nb_activations,), dtype=np.float64)
        else :
            # Copy the parent's activation mask
            self.activation_mask = self.parent.activation_mask.copy()
            # Apply the orthogonal projection
            self.activation_mask[activation_idx] = 1.0

        # Solve the LSQR system on the orthogonal projection
        # Least Squares with QR decomposition
        # A @ x = b with A = masked_conv_op, x = activations, b = signal
        masked_conv_op = self.dictionary.getMaskedConvOperator(self.activation_mask)
        activations, *_ = lsqr(masked_conv_op, signal)
        approx = masked_conv_op @ activations
        self.residual = self.signal - approx

        # Unravel the activation index to get the atom and its position
        if activation_idx is not None :
            atom_pos, atom_idx = np.unravel_index(activation_idx, shape=self.activation_mask.reshape(-1, self.nb_atoms).shape)
            self.atom_pos = atom_pos
            self.atom_idx = atom_idx
            self.atom = dictionary.getAtom(atom_idx)
            self.atom_signal = self.atom.getAtomInSignal(signal_length=self.signal_length, offset=self.atom_pos)
            self.atom_info = {'x':self.atom_pos, 'b':self.atom.b, 'y':self.atom.y, 'sigma':self.atom.sigma}

        # Initialize NetworkX ID
        self.nxid = uuid.uuid4()

    def isRoot(self) -> bool:
        return self.parent is None
    
    def isLeaf(self) -> bool:
        return len(self.children) == 0

    def getDepth(self) -> int:
        return len(np.where(self.activation_mask)[0])

    def getChildren(self) -> List['MMPNode']:
        return self.children
    
    def getAtom(self) -> ZSAtom:
        if self.isRoot() :
            return None
        return self.atom

    def getChildrenIndex(self) -> int :
        """
        Return the index of the current node in its parent's children list.
        """
        if self.isRoot() :
            return 0
        return self.parent.children.index(self)

    def computeBinaryMaskCorrelation(self, atom_idx:int, atom_pos:int, similarity:float=0.8) -> np.ndarray:
        """
        Compute a binary activation mask for the correlation with the dictionary's atoms.
        Args:
            atom_pos (int): The position of the atom in the signal
            atom_idx (int): The index of the atom in the dictionary
            similarity (float): The similarity threshold
        Returns:
            np.ndarray: The binary mask of the activation
        """
        atom_signal = self.dictionary.getAtoms()[atom_idx].getAtomInSignal(signal_length=self.signal_length, offset=atom_pos)
        correlations_with_atom = self.dictionary.computeCorrelations(atom_signal)/np.linalg.norm(atom_signal)
        return (correlations_with_atom <= similarity).astype(int)
    
    def getChildrenMaskedCorrelations(self, similarity:float=0.8) -> np.ndarray:
        signal_correlations = self.dictionary.computeCorrelations(self.residual)/np.linalg.norm(self.residual)
        masked_correlations = signal_correlations.copy()
        for child_node in self.children :
            child_pos, child_idx = child_node.atom_pos, child_node.atom_idx
            child_binary_mask = self.computeBinaryMaskCorrelation(child_idx, child_pos, similarity)
            masked_correlations *= child_binary_mask
        return masked_correlations

    def addChildNode(self, verbose=False) -> 'MMPNode':
        """
        Build a child node from the current node and its existing children.
        Returns:
            MMPNode: The child node
        """
        child_masked_correlations = self.getChildrenMaskedCorrelations()
        max_corr_idx = child_masked_correlations.argmax()
        self.children.append(MMPNode(self.dictionary, self.residual, max_corr_idx, parent=self))
        if verbose :
            print(f"Added child node with activation index {max_corr_idx}")
        return self.children[-1]
        
    def buildBranches(self, nb_branches:int) -> List['MMPNode'] :
        """
        Build the branches of the current node.
        Args:
            nb_branches (int): The number of branches to build
        Returns:
            List[MMPNode]: The list of the children nodes
        """
        while(len(self.children) < nb_branches) :
            self.addChildNode()
        return self.children
        
class MMPTree() :

    def __init__(self, dictionary:ZSDictionary, signal:np.ndarray, sparsity:int, connections:int) :
        self.dictionary = dictionary
        self.signal = signal
        self.sparsity = sparsity
        self.connections = connections
        self.root = MMPNode(dictionary, signal)
        # Initialize the tree structure
        self.init_structure()

        # Initialize the networkx graph
        self.graph = nx.DiGraph()

    def init_structure(self) :
        """
        Initialize the tree structure.
        """
        self.layers = [[self.root]]
        self.layers.extend([[] for _ in range(self.sparsity)])
        self.leaves_nodes = []
        self.leaves_paths = []

    def getCandidatePath(self, candidate_number:int) -> Tuple[int] :
        """
        Get the next path to explore in the MMP algorithm.
        """
        temp = candidate_number - 1
        path = []
        for k in range(self.sparsity) :
            path.append(temp % self.connections + 1)
            temp = temp // self.connections
        return tuple(path)

    def buildBranchFromPath(self, path:List[int], verbose=False) :
        """
        Build a branch from a path of atom indices.
        Args:
            path (List[int]): The path of atom indices
        """
        current_node = self.root
        for node_order in path :
            if node_order > len(current_node.getChildren()) :
                child_node = current_node.addChildNode(verbose=verbose)
                current_node = child_node
            else :
                current_node = current_node.getChildren()[node_order - 1]
        return current_node

    def runMMPDF(self, branches_number:int, verbose=False) :
        """
        Run the MMP algorithm with a depth-first strategy.
        """
        assert branches_number > 0, "branches_number must be greater than 0"
        assert branches_number <= self.connections ** self.sparsity, "branches_number must be less than the number of possible paths"
        # Initialize the tree structure
        self.init_structure()
        first_path = self.getCandidatePath(1)
        self.leaves_paths.append(first_path)
        # Add branches in a serial manner
        branch_counter = 0
        while len(self.leaves_paths) <= branches_number :
            # Depth-first search for the next path
            if verbose :
                print(f"Branch n°{branch_counter} exploring path : {self.leaves_paths[-1]}")
            next_path = self.getCandidatePath(len(self.leaves_paths) + 1)
            self.leaves_paths.append(next_path)
            self.leaves_nodes.append(self.buildBranchFromPath(next_path, verbose=verbose))
            branch_counter += 1

    def plotTree(self) :
        """
        Plot the tree structure.
        """
        current_layer = [(0, (0, 0))] # Tuple (node_id, (x_position, y_position))
        for layer in range(1, self.sparsity + 1) :
            next_layer = []
            width = 2 ** (self.sparsity - layer)
            for parent_id, parent_pos in current_layer :
                start_x = parent_pos[0] - (self.connections - 1) * width / 2
                for i in range(self.connections) :
                    last_node_id = parent_id * self.connections + i + 1
                    x_pos = start_x + i * width
                    self.graph.add_node(last_node_id, pos=(x_pos, -layer), mmp_node=None)
                    self.graph.add_edge(parent_id, last_node_id)
                    next_layer.append((last_node_id, (x_pos, -layer)))
            current_layer = next_layer
        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw(self.graph, pos, with_labels=False, node_color='gray', edge_color='black', node_size=700, font_size=12)
        plt.title("Generated MMP Tree")
        plt.show()