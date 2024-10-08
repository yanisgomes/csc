import time
import uuid
from tqdm import tqdm
from numba import njit, jit
import numpy as np
import pandas as pd
import networkx as nx
from einops import rearrange
from functools import partial
from itertools import product
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, lsqr
from collections import Counter

import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from joblib import Parallel, delayed

from .atoms import CSCAtom
from .utils import *

class MMPNode:
    def __init__(self, dictionary, signal:np.ndarray, dissimilarity_threshold:float=0.8, activation_idx:int=None, parent=None):
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

        # Initialize the birth time to measure the computation time
        self.birth_time = time.time()

        # //////////////<<  BEGIN COMPUTATIONAL HEART BEAT  >>\\\\\\\\\\\\\\\
        # Solve the LSQR system on the orthogonal projection
        # Least Squares with QR decomposition
        # A @ x = b with A = masked_conv_op, x = activations, b = signal
        masked_conv_op = self.dictionary.getMaskedConvOperator(self.activation_mask)
        activations, *_ = lsqr(masked_conv_op, signal)
        approx = masked_conv_op @ activations
        self.residual = self.signal - approx
        self.correlations_with_residual = self.dictionary.computeCorrelations(self.residual)/np.linalg.norm(self.residual)
        # \\\\\\\\\\\\\\\<<  END COMPUTATINAL HEART BEAT  >>/////////////////

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

        # Dissimilarity constraint threshold
        self.dissimilarity_threshold = dissimilarity_threshold

        # Compute the atom correlations
        if self.atom_signal is not None :
            self.atom_correlations = self.dictionary.computeCorrelations(self.atom_signal)/np.linalg.norm(self.atom_signal)
            self.atom_similarity_mask = self.getBinarySimilarityMask()
        else :
            self.atom_correlations = None

        # Initialize NetworkX ID
        self.nxid = uuid.uuid4()

        self.mmpdf_compute_time = self.getDelay()

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
    
    def getAtom(self) -> CSCAtom:
        if self.isRoot() :
            return None
        return self.atom
    
    def getDelay(self) -> float:
        if self.isRoot():
            return time.time() - self.birth_time
        else:
            return self.parent.getDelay()
        
    def getMMPDFComputeTime(self) -> float:
        return self.mmpdf_compute_time
    
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

    def setDissimilarityThreshold(self, threshold:float) :
        """
        Set the dissimilarity threshold of the node.
        """
        self.dissimilarity_threshold = threshold

    def computeBinaryMaskCorrelation(self, atom_idx:int, atom_pos:int) -> np.ndarray:
        """
        Compute a binary activation mask for the correlation with the dictionary's atoms.
        Args:
            atom_pos (int): The position of the atom in the signal
            atom_idx (int): The index of the atom in the dictionary
        Returns:
            np.ndarray: The binary mask of the activation
        """
        atom_signal = self.dictionary.getAtoms()[atom_idx].getAtomInSignal(signal_length=self.signal_length, offset=atom_pos)
        correlations_with_atom = self.dictionary.computeCorrelations(atom_signal)/np.linalg.norm(atom_signal)
        return (correlations_with_atom <= self.dissimilarity_threshold).astype(int)
    
    def getBinarySimilarityMask(self) -> np.ndarray:
        """
        Compute the binary similarity mask for the current node wit hitself.
        The binary similarity mask is a co-disk of the atom correlations.
        """
        return (self.atom_correlations <= self.dissimilarity_threshold).astype(int)

    def getSimilarityDisk(self) -> np.ndarray:
        """
        Compute the binary similarity disk for the current node wit hitself.
        The binary similarity disk is a disk of the atom correlations.
        """
        return (self.atom_correlations > self.dissimilarity_threshold).astype(int)
    
    def getChildrenMaskedCorrelations(self) -> np.ndarray:
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
        self.children.append(MMPNode(self.dictionary, self.residual, self.dissimilarity_threshold, max_corr_idx, parent=self))
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

    def getFullBranchPath(self) -> str:
        """
        Edit the full-blown path name of the node by concatenating parent nodes recursively.
        """
        if not self.isRoot():
            if self.parent.isRoot() :
                return f'{self.getChildrenIndex()}'
            else :
                return self.parent.getFullBranchPath() + f'-{self.getChildrenIndex()}' 
        else :
            return ''
    
class MMPTree() :

    def __init__(self, dictionary, signal:np.ndarray, sparsity:int, connections:int, dissimilarity:float=0.8): 
        self.dictionary = dictionary
        self.signal = signal
        self.sparsity = sparsity
        self.connections = connections
        self.dissimilarityMMPNode = dissimilarity
        self.root = MMPNode(dictionary, signal, self.dissimilarityMMPNode)
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

    @staticmethod
    def getCandidatePath(sparsity, connections, candidate_number:int) -> Tuple[int] :
        """
        Get the path to explore in for a given candidate number
        """
        temp = candidate_number - 1
        path = []
        for k in range(sparsity) :
            path.append(temp % connections + 1)
            temp = temp // connections
        return tuple(path)

    @staticmethod
    def getCandidateNumber(sparsity, connections, path:List[int]) -> int :   
        """
        Get the candidate number from a path of atom indices.
        """
        assert len(path) == sparsity, "The path must have the same length as the sparsity"
        candidate_number = 1
        for i, node_order in enumerate(path) :
            candidate_number += (node_order - 1) * connections ** i
        return candidate_number
    
    @staticmethod
    def getComputedNodesFromPath(path:List[int]) -> int :
        revert_path = path[::-1][:-1]
        computed_nodes = 1
        for element in revert_path :
            if element == 1 :
                computed_nodes += 1
            else :
                break
        return computed_nodes

    @staticmethod
    def getComputedNodesFromTree(sparsity:int, connections:int) -> np.ndarray :
        max_branches_number = connections**sparsity
        candidate_number = 1
        computed_nodes = list()
        while candidate_number <= max_branches_number :
            current_path = MMPTree.getCandidatePath(sparsity, connections, candidate_number)
            computed_nodes.append(MMPTree.getComputedNodesFromPath(current_path))
            candidate_number += 1
        return computed_nodes
    
    @staticmethod
    def plotComputationFromTree(sparsity: int, connections: int):
        fig = plt.figure(figsize=(10, 9))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])

        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
        sparsities = [sparsity, sparsity + 1, sparsity + 2]

        ylim_factor_list = [1.8, 2.2, 1.1]

        for ax, ylim_factor, current_sparsity in zip(axes, ylim_factor_list, sparsities):
            computations = MMPTree.getComputedNodesFromTree(current_sparsity, connections)
            max_computation = max(computations)
            nb_total_nodes = len(computations) * max_computation
            nb_computed_nodes = sum(computations)
            computation_saved_prct = 100 * (nb_total_nodes - nb_computed_nodes) / nb_total_nodes

            # Plotting
            ax.plot(computations, marker='', linestyle='-', linewidth=1.5, color='royalblue', label=f'Total computed nodes = {nb_computed_nodes}')
            ax.plot([0, len(computations) - 1], [max_computation, max_computation], 'r--', linewidth=1.5, label=f'Total nodes in the tree = {nb_total_nodes}')
            ax.fill_between(range(len(computations)), computations, color='skyblue', alpha=0.4)
            fill_between_handle = ax.fill_between(range(len(computations)), computations, max_computation, color='lightgreen', hatch='//', alpha=0.2)

            ax.set_xlabel('Branch Number')
            ax.set_ylabel('Computed Nodes')
            ax.set_title(f'Nodes Computed for K={current_sparsity} and L={connections}')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_ylim(0, max(computations) + ylim_factor)

            # Legend setup
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Patch(facecolor='lightgreen', label=f"Computation Saved = {computation_saved_prct:.2f}%", alpha=0.4))
            ax.legend(handles=handles, loc='best')
            # Ensuring only integer labels on the y-axis
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.show()


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
        #assert branches_number <= self.connections ** self.sparsity, "branches_number must be less than the number of possible paths"
        nb_branches = min(branches_number, self.connections ** self.sparsity)
        # Initialize the tree structure
        self.init_structure()
        # Add branches in a serial manner
        branch_counter = 1
        while len(self.leaves_paths) < nb_branches :
            # Depth-first search for the next path
            next_path = MMPTree.getCandidatePath(self.sparsity, self.connections, len(self.leaves_paths) + 1)
            self.leaves_paths.append(next_path)
            if verbose :
                print("\nBRANCH n°{} exploring path : {} | computed nodes = {}".format(branch_counter, self.leaves_paths[-1], MMPTree.getComputedNodesFromPath(self.leaves_paths[-1])))
            # Build the branch from the path
            self.leaves_nodes.append(self.MMPDFBranchFromPath(self.leaves_paths[-1], verbose=verbose))
            branch_counter += 1

    def runBlankMMPDF(self, branches_number:int, verbose=False) :
        """
        Simulate a run of the MMP algorithm with a depth-first strategy.
        """
        assert branches_number > 0, "branches_number must be greater than 0"
        #assert branches_number <= self.connections ** self.sparsity, "branches_number must be less than the number of possible paths"
        nb_branches = min(branches_number, self.connections ** self.sparsity)
        # Initialize the tree structure
        self.init_structure()

        total_computed_nodes = 0
        total_explored_nodes = 0

        while len(self.leaves_paths) < nb_branches :
            # Depth-first search for the next path
            next_path = MMPTree.getCandidatePath(self.sparsity, self.connections, len(self.leaves_paths) + 1)
            self.leaves_paths.append(next_path)

            if verbose :
                computed_nodes = MMPTree.getComputedNodesFromPath(self.leaves_paths[-1])
                nb_nodes = self.sparsity
                prct_computation_saved = 100 * (nb_nodes - computed_nodes) / nb_nodes
                print(f"Reused nodes proportion = {prct_computation_saved}%")

                total_computed_nodes += computed_nodes
                total_explored_nodes += nb_nodes

        if verbose :
            print(f"Total computed nodes = {total_computed_nodes}")
            print(f"Total explored nodes = {total_explored_nodes}")
            print(f"Total computation saved = {100 * (total_explored_nodes - total_computed_nodes) / total_explored_nodes}")

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
            mmp_tree_dict[str_path]['delay'] = leaf.getMMPDFComputeTime()
        return mmp_tree_dict

    def buildMMPDFResultDict(self, compute_time_type:str='local') -> Tuple[str, dict]:
        """
        Build the result item of the MMPTree :
        {'mse': min_mse, 'path': argmin_mse_path_str, 'atoms': argmin_mse_atoms, 'delay': mmpdf_compute_time}
        """
        min_mse = np.inf
        argmin_leaf = None
        argmin_mse_idx = None
        for i, leaf in enumerate(self.leaves_nodes) :
            if leaf.getMSE() <= min_mse :
                argmin_leaf = leaf
                argmin_mse_idx = i
                min_mse = leaf.getMSE()

        argmin_mse_path = self.leaves_paths[argmin_mse_idx]
        argmin_mse_path_str = '-'.join([str(p) for p in argmin_mse_path])

        if compute_time_type == 'local' :
            leaf_compute_time = argmin_leaf.getMMPDFComputeTime()
        elif compute_time_type == 'global' :
            leaf_compute_time = self.leaves_nodes[-1].getMMPDFComputeTime()

        mmpdf_result = {
            'mse' : argmin_leaf.getMSE(),
            'path' : argmin_mse_path_str,
            'atoms' : argmin_leaf.getFullBranchAtoms(),
            'delay' : leaf_compute_time
        }

        return mmpdf_result
    
    @staticmethod
    def getTreeParamsFromMMPTreeDict(mmp_tree_dict:dict) -> Tuple[int, int]:
        """
        Get the sparsity and the connections from the MMPTree dictionary.
        """
        mmp_paths = mmp_tree_dict.keys()
        sparsity = len(list(mmp_paths)[0].split('-'))
        connections = max([max([int(layer_order) for layer_order in p.split('-')]) for p in mmp_paths])
        return sparsity, connections

    @staticmethod
    def shrinkMMPTreeDict(mmp_tree_dict:dict, max_branches:int):
        """
        Shrink the MMPTree dictionary to a maximum number of branches.
        Args:
            mmp_tree_dict (dict): The MMPTree dictionary
            max_branches (int): The maximum number of branches
        Returns:
            mmp_tree_dict (dict): The shrunk MMPTree dictionary
        """
        tree_sparsity, tree_connections = MMPTree.getTreeParamsFromMMPTreeDict(mmp_tree_dict)
        keys_to_delete = []  # Initialize a list to hold the keys to be deleted
        for path, leaf_dict in mmp_tree_dict.items():
            branch_number = MMPTree.getCandidateNumber(tree_sparsity, tree_connections, list(map(int, path.split('-'))))
            if branch_number > max_branches:
                keys_to_delete.append(path)  # Add the key to the list if the branch number is too high

        # Delete the collected keys from the dictionary after the iteration
        for key in keys_to_delete:
            del mmp_tree_dict[key]

        return mmp_tree_dict
    
    @staticmethod
    def getSubTreeFromMMPTreeDict(mmp_tree_dict:dict, dictionary, signal:np.ndarray, sparsity:int, verbose:bool=False) -> dict :
        """
        Get a sub-tree from the MMPTree dictionary with a given sparsity.
        """
        tree_sparsity, tree_connections = MMPTree.getTreeParamsFromMMPTreeDict(mmp_tree_dict)
        mmp_sub_tree_dict = {}
        # Iterate over the MMPTree branches
        for path, leaf_dict in mmp_tree_dict.items() :
            # Get the sub-path according to the sparsity
            path_list = list(map(int, path.split('-')))
            sub_path_list = path_list[:sparsity]
            sub_path = '-'.join([str(p) for p in sub_path_list])
            if verbose :
                print(f'')
            if sub_path not in mmp_sub_tree_dict.keys() :

                # Get the atoms of the sub-path
                sub_path_atoms = leaf_dict['atoms'][:sparsity]

                # Build the approximation of the sub-path
                sub_path_approx, _ = dictionary.getSignalProjectionFromAtoms(signal, sub_path_atoms)
                sub_path_mse = np.mean((signal - sub_path_approx) ** 2)

                mmp_sub_tree_dict[sub_path] = {'atoms':sub_path_atoms, 'mse':sub_path_mse}
                if verbose :
                    print(f'Sub-path {sub_path} : MSE = {sub_path_mse}')

        return mmp_sub_tree_dict
    
    @staticmethod
    def mmpdfCandidateFromMMPTreeDict(mmp_tree_dict:dict, dictionary, signal:np.ndarray, candidate_sparsity:int, verbose:bool=False) :
        """
        Build a candidate from the MMPTree dictionary if the MMP algorithm has been run with candidate_sparsity.
        Args:
            mmp_tree_dict (dict): The MMPTree dictionary
            atom_length (int): The length of the atoms
            signal (np.ndarray): The signal to approximate
            candidate_sparsity (int): The sparsity of the candidate
            verbose (bool): The verbosity flag
        Returns:
            candidate_atoms (list): The candidate atoms
        """
        tree_sparsity, tree_connections = MMPTree.getTreeParamsFromMMPTreeDict(mmp_tree_dict)

        # The subtree is the MMPTree result if the sparsity had been candidate_sparsity < tree_sparsity
        mmp_sub_tree_dict = MMPTree.getSubTreeFromMMPTreeDict(mmp_tree_dict, dictionary, signal, candidate_sparsity)

        max_sparsity = min(candidate_sparsity, tree_sparsity)
        min_mse = np.inf
        argmin_mse_branch = '-'.join(['1' for _ in range(max_sparsity)])

        for path, leaf_dict in mmp_sub_tree_dict.items() :
            if leaf_dict['mse'] < min_mse :
                min_mse = leaf_dict['mse']
                argmin_mse_branch = path

        argmin_mse_atoms = mmp_sub_tree_dict[argmin_mse_branch]['atoms']
        
        if verbose :
            print(f'MMP-DF : {argmin_mse_branch} | MSE = {min_mse}')

        return argmin_mse_atoms, min_mse
            
    @staticmethod
    def ompCandidateFromMMPTreeDict(mmp_tree_dict:dict, dictionary, signal:np.ndarray, candidate_sparsity:int, verbose:bool=False) :
        """
        Build a candidate from the MMPTree dictionary if the OMP algorithm has been run with candidate_sparsity.
        Args:
            mmp_tree_dict (dict): The MMPTree dictionary
            signal (np.ndarray): The signal to approximate
            candidate_sparsity (int): The sparsity of the candidate
        Returns:
            candidate_atoms (list): The candidate atoms
        """
        tree_sparsity, tree_connections = MMPTree.getTreeParamsFromMMPTreeDict(mmp_tree_dict)
        
        omp_branch_name = '-'.join(['1' for _ in range(tree_sparsity)])
        omp_branch_dict = mmp_tree_dict[omp_branch_name]
        omp_atoms = omp_branch_dict['atoms']

        # Get the sub branch
        omp_sub_branch_name = '-'.join(['1' for _ in range(min(candidate_sparsity, tree_sparsity))])
        omp_sub_branch_atoms = omp_atoms[:min(candidate_sparsity, tree_sparsity)]

        # Build the candidate signal
        candidate_approx, _ = dictionary.getSignalProjectionFromAtoms(signal, omp_sub_branch_atoms)
        candidate_mse = np.mean((signal - candidate_approx) ** 2)

        if verbose :
            print(f'OMP : {omp_sub_branch_name} MSE = {candidate_mse}')

        return omp_sub_branch_atoms, candidate_mse