import os
import time
import json
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Union
from scipy import optimize

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
    def dictFromId(db_path:str, key:str, id:int) -> Dict:
        """
        Get the signal dictionary from its ID.
        Args:
            db_path (str): Path to the database.
            id (int): Signal ID.
        Returns:
            dict: Signal dictionary.
        """
        with open(db_path, 'r') as f:
            output_data = json.load(f)
            signals_dict= next((item for item in output_data[key] if item['id'] == id), None)
        return signals_dict

    @staticmethod
    def dictsFromIds(db_path:str, key:str, ids:List[int]) -> List[Dict]:
        """
        Get the signal dictionary from its ID.
        Args:
            db_path (str): Path to the database.
            id (int): Signal ID.
        Returns:
            dict: Signal dictionary.
        """
        with open(db_path, 'r') as f:
            output_data = json.load(f)
            signals_dict= next((item for item in output_data[key] if item['id'] in ids), None)
        return signals_dict
    
    @staticmethod
    def positionMatching(true_atoms:List[Dict], approx_atoms:List[Dict]) -> List[Tuple[Dict,Dict]]:
        """
        Compute the best position matching using the hungarian algorithm
        Args:
            true_atoms (List[Dict]): Atoms of the true dictionary.
            approx_atoms (List[Dict]): Atoms of the approximation dictionary.
        Returns:
            List[Tuple[Dict,Dict]]: List of tuples of matched atoms dict.
        """
        true_positions = np.array([atom['x'] for atom in true_atoms])
        approx_positions = np.array([atom['x'] for atom in approx_atoms])
        cost_matrix = np.abs(true_positions[:, np.newaxis] - approx_positions)
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
        matched_atoms = [(true_atoms[i], approx_atoms[j]) for i, j in zip(row_ind, col_ind)]
        return matched_atoms
    
    @staticmethod
    def meanPositionError(true_atoms:List[Dict], approx_atoms:List[Dict]) -> List[int]:
        """
        Compute the position error between the true and approximation dictionaries.
        Args:
            true_atoms (List[Dict]): Atoms of the true dictionary.
            approx_atoms (List[Dict]): Atoms of the approximation dictionary.
        Returns:
            List[int]: List of position errors.
        """
        positions_errors = [true_atom['x'] - approx_atom['x'] for true_atom, approx_atom in CSCWorkbench.positionMatching(true_atoms, approx_atoms)]
        return np.mean(positions_errors)
    
    @staticmethod
    def positionErrorPerStep(true_atoms:List[Dict], approx_atoms:List[Dict]) -> List[int]:
        """
        Compute the position error between the true and approximation dictionaries.
        Args:
            true_atoms (List[Dict]): Atoms of the true dictionary.
            approx_atoms (List[Dict]): Atoms of the approximation dictionary.
        Returns:
            List[int]: List of position errors.
        """
        positions_errors = [true_atom['x'] - approx_atom['x'] for true_atom, approx_atom in CSCWorkbench.positionMatching(true_atoms, approx_atoms)]
        return positions_errors
    
    def computeMeanPositionErrors(self, db_path:str) -> Dict:
        """
        Compute the position errors for all the signals.
        """
        # Load the data
        with open(db_path, 'r') as f:
            output_data = json.load(f)

        data_errors = {
            'id': [],
            'snr': [],
            'sparsity': [],
            'pos_err': []
        }
        # Iterate over the outputs
        for result in output_data['omp'] :
            # Get signal and approximation atoms
            signal_id = result['id']
            signal_dict = self.signalDictFromId(signal_id)
            signal_atoms = signal_dict['atoms']
            approx_atoms = result['atoms']
            # Compute errors
            mean_pos_error = CSCWorkbench.meanPositionError(signal_atoms, approx_atoms)
            # Append data
            data_errors['id'].append(signal_id)
            data_errors['snr'].append(signal_dict['snr'])
            data_errors['sparsity'].append(signal_dict['sparsity'])
            data_errors['pos_err'].append(mean_pos_error)
        return data_errors
    
    def computePositionErrorsPerStep(self, db_path:str) -> Dict:
        """
        Compute the position errors for all the signals.
        """
        # Load the data
        with open(db_path, 'r') as f:
            output_data = json.load(f)

        data_errors = {
            'id': [],
            'snr': [],
            'sparsity': [],
            'pos_err': [],
            'abs_pos_err': [],
            'algo_step': []
        }
        # Iterate over the outputs
        for result in output_data['omp'] :
            # Get signal and approximation atoms
            signal_id = result['id']
            signal_dict = self.signalDictFromId(signal_id)
            signal_atoms = signal_dict['atoms']
            approx_atoms = result['atoms']
            # Compute errors
            pos_error_per_step = CSCWorkbench.positionErrorPerStep(signal_atoms, approx_atoms)
            # Append data
            for i, err in enumerate(pos_error_per_step):
                data_errors['id'].append(signal_id)
                data_errors['snr'].append(signal_dict['snr'])
                data_errors['sparsity'].append(signal_dict['sparsity'])
                data_errors['pos_err'].append(err)
                data_errors['abs_pos_err'].append(np.abs(err))
                data_errors['algo_step'].append(i+1)
        return data_errors
    
    def plotMeanPosErr(self, db_path:str) :
        """
        Plot the boxplot of the position errors.
        """
        plt.figure(figsize=(12, 8)) 
        data_errors = self.computeMeanPositionErrors(db_path)
        df = pd.DataFrame(data_errors)
        #sns.boxplot(x='snr', y='pos_err', hue='sparsity', data=df)
        #sns.color_palette("flare", as_cmap=True)
        sns.color_palette("crest", as_cmap=True)
        sns.boxplot(x='snr', y='pos_err', hue='sparsity', data=df, palette="flare", fliersize=2, whis=1.5, showfliers=False)
        sns.despine(trim=True)
        #plt.yscale('log')  # Échelle logarithmique pour l'axe des y
        plt.title('OMP boxplot of Position Errors by SNR and Sparsity', fontsize=14)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=12)
        plt.ylabel('Position Error', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Sparsity', loc='best')
        plt.show()

    def plotPosErrAtSparsity(self, db_path:str, sparsity:int) :
        """
        Plot the boxplot of the position errors.
        """
        plt.figure(figsize=(12, 8)) 
        data_errors = self.computePositionErrorsPerStep(db_path)
        df_all_sparsities = pd.DataFrame(data_errors)
        df = df_all_sparsities.loc[df_all_sparsities['sparsity'] == sparsity]
        sns.boxplot(x='snr', y='pos_err', hue='algo_step', data=df, palette="flare", fliersize=2, whis=1.5, showfliers=False)
        sns.despine(trim=True)
        plt.title(f'Position errors by SNR and OMP step at sparsity = {sparsity}', fontsize=14)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=12)
        plt.ylabel('Position Error', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='OMP Step', loc='best')
        plt.show()

    def plotPosErrAtStep(self, db_path:str, step:int) :
        """
        Plot the boxplot of the position errors.
        """
        plt.figure(figsize=(12, 8)) 
        data_errors = self.computePositionErrorsPerStep(db_path)
        df_all_steps = pd.DataFrame(data_errors)
        df = df_all_steps.loc[df_all_steps['algo_step'] == step]
        sns.boxplot(x='snr', y='pos_err', hue='sparsity', data=df, palette="flare", fliersize=2, whis=1.5, showfliers=False)
        sns.despine(trim=True)
        plt.title(f'Position errors by SNR and sparsity at step = {step}', fontsize=14)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=12)
        plt.ylabel('Sparsity', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Sparsity', loc='best')
        plt.show()
        
    def sortByPositionErrorAtStep(self, db_path:str, step:int, ascending:bool=True) :
        """
        Plot the boxplot of the position errors.
        """
        data_errors = self.computePositionErrorsPerStep(db_path)
        df_all_steps = pd.DataFrame(data_errors)
        df = df_all_steps.loc[df_all_steps['algo_step'] == step]
        df = df.sort_values(by='abs_pos_err', ascending=ascending)
        return df
    
    def plotComparison(self, db_path:str, ids:List[int]) -> None :
        """
        Plot the signal decomposition.
        Args:
            signal_dict (Dict): Signal dictionary.
        """
        # Load the data
        with open(db_path, 'r') as f:
            output_data = json.load(f)
            approxs_dict = [result for id in ids for result in output_data['omp'] if result['id'] == id]
        # Plot the comparison
        fig, axs = plt.subplots(len(ids), 1, figsize=(12, 4*len(ids)), sharex=True)
        for i, approx_dict in enumerate(approxs_dict):
            # Plot the noisy signal
            signal_dict = self.signalDictFromId(approx_dict['id'])
            axs[i].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
            # Recover the true signal and plot it
            true_signal = np.zeros_like(signal_dict['signal'])
            for atom in signal_dict['atoms']:
                zs_atom = ZSAtom.from_dict(atom)
                zs_atom.padBothSides(self.dictionary.getAtomsLength())
                true_signal += zs_atom.getAtomInSignal(len(true_signal), atom['x'])
            axs[i].plot(true_signal, label='True signal', color='g', alpha=0.9, lw=2)
            axs[i].plot(approx_dict['approx'], label='OMP Reconstruction')
            axs[i].set_title(f"Decomposition of signal n°{approx_dict['id']}")
            axs[i].legend(loc='best')
        plt.show()

    def plotStepDecomposition(self, db_path:str, id:int) -> None :
        """
        Plot the signal decomposition.
        Args:
            signal_dict (Dict): Signal dictionary.
        """
        # Load the data
        with open(db_path, 'r') as f:
            output_data = json.load(f)
            approx_dict = next((result for result in output_data['omp'] if result['id'] == id), None)
        # Get the true signal
        signal_dict = self.signalDictFromId(id)
        # Plot the comparison
        fig, axs = plt.subplots(len(approx_dict['atoms'])+1, 1, figsize=(12, 4*len(approx_dict['atoms'])), sharex=True)
    
        trueSuperposition = np.zeros_like(signal_dict['signal'])
        approxSuperposition = np.zeros_like(signal_dict['signal'])
        
        for i, (true_atom_dict, approx_atom_dict) in enumerate(zip(signal_dict['atoms'], approx_dict['atoms'])):
            # Construct the atoms from parameters
            true_atom = ZSAtom.from_dict(true_atom_dict)
            true_atom.padBothSides(self.dictionary.getAtomsLength())
            approx_atom = ZSAtom.from_dict(approx_atom_dict)
            approx_atom.padBothSides(self.dictionary.getAtomsLength())
            # Get the atom signals
            true_atom_signal = true_atom.getAtomInSignal(len(signal_dict['signal']), true_atom_dict['x'])
            approx_atom_signal = approx_atom.getAtomInSignal(len(signal_dict['signal']), approx_atom_dict['x'])
            # Apply the atoms superposition
            trueSuperposition += true_atom_signal
            approxSuperposition += approx_atom_signal
            # Plot the atoms and the noisy signal
            axs[i+1].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.3, lw=3)
            axs[i+1].plot(true_atom_signal, label='True atom', alpha=0.6, lw=2)
            axs[i+1].plot(approx_atom_signal, label='Approx atom', alpha=0.9, lw=1)
            axs[i+1].set_title(f'Step n°{i+1}')
            axs[i+1].legend(loc='best')

        axs[0].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
        axs[0].plot(trueSuperposition, label='True superposition', color='g', alpha=0.9, lw=2)
        axs[0].plot(approxSuperposition, label='Approx superposition')
        axs[0].set_title('Signal n°{} decomposition : MSE = {:.2e}'.format(id, approx_dict['mse']))
        axs[0].legend(loc='best')
        plt.show()