import os
import time
import json
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Union
from scipy.optimize import linear_sum_assignment 

from .dictionary import ZSDictionary
from .atoms import ZSAtom
from .utils import *

class CSCWorkbench:

    def __init__(self, signals_path:str):

        # Signals data
        self.signals_path = signals_path
        self.signals_data = None

        # Signals parameters
        self.signals_length = None
        self.batch_size = None
        self.snr_levels = None
        self.sparsity_levels = None

        # Workbench state
        self.loaded = False

        # ZSDictionary
        self.dictionary = None

    def load_data(self):
        """
        Charge data from the specified file path using pandas.
        """
        with open(self.signals_path, 'r') as f:
            self.signals_data = json.load(f)
            self.signals_length = self.signals_data['signalLength']
            self.batch_size = self.signals_data['batchSize']
            self.snr_levels = self.signals_data['snrLevels']
            self.sparsity_levels = self.signals_data['sparsityLevels']
            self.loaded = True

    def set_dictionary(self, dictionary: ZSDictionary):
        """
        Charge a dictionary to the workbench.
        """
        self.dictionary = dictionary

    def signalDictFromId(self, id:int) -> Dict:
        """
        Get the signal dictionary from its ID.
        Args:
            id (int): Signal ID.
        Returns:
            dict: Signal dictionary.
        """
        if not self.loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        dict_signal = next((item for item in self.signals_data['signals'] if item['id'] == id), None)
        return dict_signal
    
    @staticmethod
    def positionMatching(true_atoms:List[Dict], approx_atoms:List[Dict]) -> List[Tuple[ZSAtom,ZSAtom]]:
        """
        Match the atoms of the true and approximation dictionaries.
        Args:
            true_atoms (List[ZSAtom]): Atoms of the true dictionary.
            approx_atoms (List[ZSAtom]): Atoms of the approximation dictionary.
        Returns:
            List[Tuple[ZSAtom,ZSAtom]]: List of tuples with the matching atoms.
        """
        matched_atoms = []
        # Associate each true atom with the closest approximation atom
        for true_atom in true_atoms:
            closest_atom = min(approx_atoms, key=lambda approx: abs(approx['x'] - true_atom['x']))
            matched_atoms.append((true_atom, closest_atom))
        return matched_atoms

    @staticmethod
    def positionMatching(true_atoms:List[Dict], approx_atoms:List[Dict]) -> List[Tuple[ZSAtom,ZSAtom]]:
        """
        Match the atoms of the true and approximation dictionaries using the Hungarian algorithm.
        Args:
            true_atoms (List[ZSAtom]): Atoms of the true dictionary.
            approx_atoms (List[ZSAtom]): Atoms of the approximation dictionary.
        Returns:
            List[Tuple[ZSAtom,ZSAtom]]: List of tuples with the matching atoms.
        """
        true_positions = np.array([atom['x'] for atom in true_atoms])
        approx_positions = np.array([atom['x'] for atom in approx_atoms])
        cost_matrix = np.abs(true_positions[:, np.newaxis] - approx_positions)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_atoms = [(true_atoms[i], approx_atoms[j]) for i, j in zip(row_ind, col_ind)]
        return matched_atoms
    
    
    @staticmethod
    def positionError(true_atoms:List[Dict], approx_atoms:List[Dict]) -> List[int]:
        """
        Compute the position error between the true and approximation dictionaries.
        Args:
            true_atoms (List[Dict]): Atoms of the true dictionary.
            approx_atoms (List[Dict]): Atoms of the approximation dictionary.
        Returns:
            List[int]: List of position errors.
        """
        return [abs(true_atom['x'] - approx_atom['x']) for true_atom, approx_atom in CSCWorkbench.positionMatching(true_atoms, approx_atoms)]
    
    def computePositionErrors(self, db_path:str) -> Dict:
        """
        Compute the position errors for all the signals.
        """
        # Load the data
        with open(db_path, 'r') as f:
            output_data = json.load(f)

        data_errors = {
            'snr': [],
            'sparsity': [],
            'position_error': []
        }
        print('signals_data :', len(self.signals_data['signals']))
        print('output_data :', len(output_data['omp']))
        # Iterate over the outputs
        for result in output_data['omp'] :
            # Get signal and approximation atoms
            signal_id = result['id']
            signal_dict = self.signalDictFromId(signal_id)
            signal_atoms = signal_dict['atoms']
            approx_atoms = result['atoms']
            # Compute errors
            errors = CSCWorkbench.positionError(signal_atoms, approx_atoms)
            # Append data
            for error in errors:
                data_errors['snr'].append(signal_dict['snr'])
                data_errors['sparsity'].append(signal_dict['sparsity'])
                data_errors['position_error'].append(error)

        print('data errors :')
        print('snr : ', len(data_errors['snr']))
        print('sparsity : ', len(data_errors['sparsity']))
        print('position_error : ', len(data_errors['position_error']))
        return data_errors
    
    def boxplotPositionErrors(self, db_path:str) :
        """
        Plot the boxplot of the position errors.
        """
        plt.figure(figsize=(12, 8)) 
        data_errors = self.computePositionErrors(db_path)
        df = pd.DataFrame(data_errors)
        #sns.boxplot(x='snr', y='position_error', hue='sparsity', data=df)
        #sns.color_palette("flare", as_cmap=True)
        sns.color_palette("crest", as_cmap=True)
        sns.boxplot(x='snr', y='position_error', hue='sparsity', data=df, palette="flare", fliersize=2, whis=1.5, showfliers=False)
        sns.despine(trim=True)
        #plt.yscale('log')  # Échelle logarithmique pour l'axe des y
        plt.title('OMP boxplot of Position Errors by SNR and Sparsity', fontsize=14)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=12)
        plt.ylabel('Position Error', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Sparsity', loc='best')
        plt.show()

    def violinplotPositionErrors(self, db_path:str) :
        """
        Plot the boxplot of the position errors.
        """
        plt.figure(figsize=(12, 8)) 
        data_errors = self.computePositionErrors(db_path)
        df = pd.DataFrame(data_errors)
        #sns.boxplot(x='snr', y='position_error', hue='sparsity', data=df)
        #sns.color_palette("flare", as_cmap=True)
        sns.color_palette("crest", as_cmap=True)
        sns.violinplot(x='snr', y='position_error', hue='sparsity', data=df, inner="box", palette="crest", alpha=0.6, cut=0)
        sns.despine(trim=True)
        #plt.yscale('log')  # Échelle logarithmique pour l'axe des y
        plt.title('OMP violinplot of Position Errors by SNR and Sparsity', fontsize=14)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=12)
        plt.ylabel('Position Error', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Sparsity', loc='best')
        plt.show()
