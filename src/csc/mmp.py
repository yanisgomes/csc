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

from .atoms import ZSAtom
from .utils import *

class MMPNode:
    def __init__(self, dictionary, signal:np.ndarray, activation_idx:int=None, parent=None):
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

        # //////////////<<  BEGIN COMPUTATINAL HEART BEAT  >>\\\\\\\\\\\\\\\
        # Solve the LSQR system on the orthogonal projection
        # Least Squares with QR decomposition
        # A @ x = b with A = masked_conv_op, x = activations, b = signal
        masked_conv_op = self.dictionary.getMaskedConvOperator(self.activation_mask)
        activations, *_ = lsqr(masked_conv_op, signal)
        approx = masked_conv_op @ activations
        self.residual = self.signal - approx
        self.correlations_with_residual = self.dictionary.computeCorrelations(self.residual)/np.linalg.norm(self.residual)
        # \\\\\\\\\\\\\\\<<  END COMPUTATINAL HEART BEAT  >>///////////////

        # Unravel the activation index to get the atom and its position
        if activation_idx is not None :
            atom_pos, atom_idx = np.unravel_index(activation_idx, shape=self.activation_mask.reshape(-1, self.nb_atoms).shape)
            self.atom_pos = atom_pos
            self.atom_idx = atom_idx
            self.atom = dictionary.getAtom(atom_idx)
            self.atom_signal = self.atom.getAtomInSignal(signal_length=self.signal_length, offset=self.atom_pos)
            self.atom_info = {'x':self.atom_pos, 'b':self.atom.b, 'y':self.atom.y, 's':self.atom.sigma}
        else :
            self.atom_pos = None
            self.atom_idx = None
            self.atom = None
            self.atom_signal = None
            self.atom_info = None

        # Compute the atom correlations
        if self.atom_signal is not None :
            self.atom_correlations = self.dictionary.computeCorrelations(self.atom_signal)/np.linalg.norm(self.atom_signal)
            self.atom_similarity_mask = self.getBinarySimilarityMask()
        else :
            self.atom_correlations = None

        # Initialize NetworkX ID
        self.nxid = uuid.uuid4()

    def isRoot(self) -> bool:
        return self.parent is None
    
    def isLeaf(self) -> bool:
        return len(self.children) == 0
    
    def getMSE(self) -> float:
        return np.mean(self.residual ** 2)

    def getDepth(self) -> int:
        return len(np.where(self.activation_mask)[0])

    def getChildren(self) -> List['MMPNode']:
        return self.children
    
    def getAtom(self) -> ZSAtom:
        if self.isRoot() :
            return None
        return self.atom
    
    def getAtomSignal(self) -> np.ndarray:
        if self.isRoot() :
            return None
        return self.atom_signal
    
    def getGenealogy(self) -> List['MMPNode']:
        """
        Return the genealogy of the current node.
        """
        genealogy = list()
        current_node = self
        while not current_node.isRoot() :
            genealogy.append(current_node)
            current_node = current_node.parent
        return genealogy

    def getChildrenIndex(self) -> int :
        """
        Return the index of the current node in its parent's children list.
        """
        if self.isRoot() :
            return 0
        return self.parent.children.index(self) + 1

    def getGenealogyIndex(self) -> List[int]:
        """
        Return the genealogy index of the current node.
        """
        genealogy_index = list()
        current_node = self
        while not current_node.isRoot() :
            genealogy_index.append(current_node.getChildrenIndex())
            current_node = current_node.parent
        genealogy_index.reverse()
        return genealogy_index
    
    def getGenealogyStr(self) -> List['MMPNode']:
        """
        Return the genealogy of the current node.
        """
        genealogy_str = 'ROOT'
        if not self.isRoot() :
            for idx in self.getGenealogyIndex() :   
                genealogy_str += f' -> {idx}'
        return genealogy_str

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
    
    def getBinarySimilarityMask(self, similarity:float=0.8) -> np.ndarray:
        """
        Compute the binary similarity mask for the current node wit hitself.
        """
        return (self.atom_correlations <= similarity).astype(int)
    
    def getChildrenMaskedCorrelations(self, similarity:float=0.8) -> np.ndarray:
        signal_correlations = self.dictionary.computeCorrelations(self.residual)/np.linalg.norm(self.residual)
        masked_correlations = signal_correlations.copy()
        for child_node in self.children :
            child_pos, child_idx = child_node.atom_pos, child_node.atom_idx
            child_binary_mask = self.computeBinaryMaskCorrelation(child_idx, child_pos, similarity)
            masked_correlations *= child_binary_mask
        return masked_correlations
    
    def getChildrenMaskedCorrelations(self, similarity:float=0.8) -> np.ndarray:
        """
        Compute the masked correlations of the children nodes.
        """
        masked_correlations = self.correlations_with_residual.copy()
        for child_node in self.children :
            masked_correlations *= child_node.atom_similarity_mask
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
            print(f"    NEW MMPNode : {self.getGenealogyStr()}  +  Node({len(self.children)})")
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
    
    def getResidual(self) -> np.ndarray:
        return self.residual
    
    def buildSignalRecovery(self) -> np.ndarray:
        """
        Build the approximation of the signal from the root to the current node.
        """
        if self.isRoot() :
            return np.zeros_like(self.signal)
        else :
            return self.parent.buildSignalRecovery() + self.atom_signal
        
    def getFullBranchAtoms(self) -> dict:
        """
        Edit the full-blown dictionary of the node by concatenating atom_info dictionaries
        from parent nodes recursively.
        """
        if not self.isRoot():
            return self.parent.getFullBranchAtoms() + [self.atom_info]  
        else :
            return []
    
class MMPTree() :

    def __init__(self, dictionary, signal:np.ndarray, sparsity:int, connections:int) :
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

    def getCandidateNumber(self, path:List[int]) -> int :   
        """
        Get the candidate number from a path of atom indices.
        """
        assert len(path) == self.sparsity, "The path must have the same length as the sparsity"
        candidate_number = 1
        for i, node_order in enumerate(path) :
            candidate_number += (node_order - 1) * self.connections ** i
        return candidate_number

    def MMPDFBranchFromPath(self, path:List[int], verbose=False) :
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
                if verbose :
                    print(f"   ~~ Using MMPNode ({current_node.getGenealogyStr()}) ~~")
        return current_node

    def runMMPDF(self, branches_number:int, verbose=False) :
        """
        Run the MMP algorithm with a depth-first strategy.
        """
        assert branches_number > 0, "branches_number must be greater than 0"
        assert branches_number <= self.connections ** self.sparsity, "branches_number must be less than the number of possible paths"
        # Initialize the tree structure
        self.init_structure()
        # Add branches in a serial manner
        branch_counter = 1
        while len(self.leaves_paths) < branches_number :
            # Depth-first search for the next path
            next_path = self.getCandidatePath(len(self.leaves_paths) + 1)
            self.leaves_paths.append(next_path)
            if verbose :
                print(f"\nBRANCH n°{branch_counter} exploring path : {self.leaves_paths[-1]}")
            # Build the branch from the path
            self.leaves_nodes.append(self.MMPDFBranchFromPath(self.leaves_paths[-1], verbose=verbose))
            branch_counter += 1

    def getResult(self) -> Tuple[np.ndarray, List[dict]]:
        """
        Get the approximation of the signal from the leaves.
        Returns:
            Tuple[np.ndarray, List[dict]]: The approximation and the atoms info
        """
        argmin_mse = np.argmin([leaf.getMSE() for leaf in self.leaves_nodes])
        approx = self.leaves_nodes[argmin_mse].buildSignalRecovery()
        infos = self.leaves_nodes[argmin_mse].getFullBranchAtoms()
        return approx, infos

    def printLeaves(self) :
        """
        Print the leaves of the tree.
        """
        for i, leaf in enumerate(self.leaves_nodes) :
            print(f'Branch n°{i+1} leaf with MSE = {leaf.getMSE()}')

    def plotLeavesComparison(self) :
        """
        Plot the comparison of the leaves of the tree.
        """
        # Build the figure
        nb_leaves = len(self.leaves_nodes)
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # First plot all the leaves approximations
        axs[0].plot(self.signal, label='Original signal', color='k', alpha=0.4, lw=3)
        argmin_mse = np.argmin([leaf.getMSE() for leaf in self.leaves_nodes])
        for i, leaf in enumerate(self.leaves_nodes) :
            axs[0].plot(leaf.buildSignalRecovery(), label='Leaf n°{} : MSE = {:.2e}'.format(i, leaf.getMSE()))
        axs[0].set_title('Comparison of the leaves approximations')
        axs[0].legend(loc='best')
        axs[0].axis('off')

        # Plot the atom signals of the best leaf
        axs[1].plot(self.signal, label='Original signal', color='k', alpha=0.4, lw=3)
        for node in self.leaves_nodes[argmin_mse].getGenealogy() :
            axs[1].plot(node.getAtomSignal(), label=f'{str(node.atom)}')
        axs[1].set_title('Atom decomposition for the argmin(MSE) leaf')
        axs[1].legend(loc='best')
        axs[1].axis('off')

        plt.show()

    def plotLeafDecomposition(self, leaf_idx:int) :
        """
        Plot the decomposition of a leaf.
        """
        leaf = self.leaves_nodes[leaf_idx]
        leaf_genealogy = leaf.getGenealogy()
        leaf_genealogy.reverse()

        fig, axs = plt.subplots(len(leaf_genealogy)+1, 1, figsize=(12, 3*len(leaf_genealogy)+1))
        
        axs[0].plot(self.signal, label='Original signal', color='k', alpha=0.4, lw=3)
        axs[0].plot(leaf.buildSignalRecovery(), label='Approximation')
        axs[0].legend()
        axs[0].axis('off')

        for i, leaf in enumerate(leaf_genealogy) :
            axs[i+1].plot(leaf.signal, label="Local leaf's residual", color='k', alpha=0.4, lw=3)
            axs[i+1].plot(leaf.getAtomSignal(), label=f'{str(leaf.atom)}')
            axs[i+1].legend()
            axs[i+1].axis('off')

        plt.show()

    def plotLeavesComparisonFromIdx(self, idx1:int, idx2:int) :
        """
        Plot the comparison between two leaves.
        """
        leaf1 = self.leaves_nodes[idx1]
        leaf2 = self.leaves_nodes[idx2]

        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        axs[0].plot(self.signal, label='Original signal', color='k', alpha=0.4, lw=3)
        axs[0].plot(leaf1.buildSignalRecovery(), label='Leaf n°{}'.format(idx1))
        axs[0].plot(leaf2.buildSignalRecovery(), label='Leaf n°{}'.format(idx2))
        axs[0].set_title('Comparison of the approximations')
        axs[0].legend(loc='best')
        axs[0].axis('off')

        axs[1].plot(self.signal, label='Original signal', color='k', alpha=0.4, lw=3)
        for node in leaf1.getGenealogy() :
            axs[1].plot(node.getAtomSignal(), label=f'{str(node.atom)}')
        axs[1].set_title('Atom decomposition for leaf n°{}'.format(idx1))
        axs[1].legend(loc='best')
        axs[1].axis('off')

        axs[2].plot(self.signal, label='Original signal', color='k', alpha=0.4, lw=3)
        for node in leaf2.getGenealogy() :
            axs[2].plot(node.getAtomSignal(), label=f'{str(node.atom)}')
        axs[2].set_title('Atom decomposition for leaf n°{}'.format(idx2))
        axs[2].legend(loc='best')
        axs[2].axis('off')

        plt.show()

    def plotOMPComparison(self) :
        """
        Plot the comparison between the OMP and the MMP algorithms.
        """
        self.plotLeavesComparisonFromIdx(0, np.argmin([leaf.getMSE() for leaf in self.leaves_nodes]))

    def buildMMPTreeDict(self) -> Dict:
        """
        Build the dictionary of the MMPTree.
        """
        mmp_tree_dict = {}
        for path, leaf in zip(self.leaves_paths, self.leaves_nodes) :
            str_path = '-'.join([str(p) for p in path])
            mmp_tree_dict[str_path] = dict()
            mmp_tree_dict[str_path]['mse'] = leaf.getMSE()
            mmp_tree_dict[str_path]['atoms'] = leaf.getFullBranchAtoms()
        return mmp_tree_dict
    
    