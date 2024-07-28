import os
import time
import json
import math
import itertools
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List, Dict, Tuple, Any

from scipy import optimize
import scipy.stats as stats
import scikit_posthocs as sp
from scipy.interpolate import interp1d

from .dictionary import ZSDictionary
from .atoms import ZSAtom
from .mmp import MMPTree
from .utils import *

from joblib import Parallel, delayed

from alphacsc.update_z import update_z

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from critdd import Diagram


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

        # F1 score threshold
        self.f1CorrThreshold = 0.9

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

    @staticmethod
    def loadDataFromPath(db_path):
        with open(db_path, 'r') as file:
            return json.load(file)

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

    def computeMatchingPosition(self, true_atoms:List[Dict], approx_atoms:List[Dict]) -> List[Tuple[Dict,Dict]]:
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

    def computeMatchingCorrelation(self, true_atoms_dict:List[Dict], approx_atoms_dict:List[Dict]) -> List[Tuple[Dict,Dict]]:
        """
        Compute the best correlation matching using the hungarian algorithm
        Args:
            true_atoms (List[Dict]): Atoms of the true dictionary.
            approx_atoms (List[Dict]): Atoms of the approximation dictionary.
        Returns:
            List[Tuple[Dict,Dict]]: List of tuples of matched atoms dict.
        """

        true_atoms = [ZSAtom.from_dict(atom) for atom in true_atoms_dict]
        approx_atoms = [ZSAtom.from_dict(atom) for atom in approx_atoms_dict]
        # Pad the atoms
        for atom in true_atoms:
            atom.padBothSides(self.dictionary.getAtomsLength())
        for atom in approx_atoms:
            atom.padBothSides(self.dictionary.getAtomsLength())
        # Compute the signals
        true_atom_signals = [atom.getAtomInSignal(self.signals_length, atom_dict['x']) for atom, atom_dict in zip(true_atoms, true_atoms_dict)]
        approx_atom_signals = [atom.getAtomInSignal(self.signals_length, atom_dict['x']) for atom, atom_dict in zip(approx_atoms, approx_atoms_dict)]
    
        # Compute the cost matrix as the MSE between each pair of true and approx atom signals
        cost_matrix = np.zeros((len(true_atoms), len(approx_atoms)))
        for i, true_signal in enumerate(true_atom_signals):
            for j, approx_signal in enumerate(approx_atom_signals):
                cost_matrix[i, j] = abs(np.correlate(true_signal, approx_signal, mode='valid')[0])

        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
        matched_atoms = [(true_atoms_dict[i], approx_atoms_dict[j]) for i, j in zip(row_ind, col_ind)]
        return matched_atoms

    def computeMaxTruePositives(self, true_atoms:List[Dict], approx_atoms:List[Dict], pos_err_threshold:float, corr_err_threshold:float, verbose:bool=False) -> int:
        """
        Compute the maximum number of true positives using the Hungarian algorithm.
        Args:
            true_atoms (List[Dict]): Atoms of the true dictionary.
            approx_atoms (List[Dict]): Atoms of the approximation dictionary.
            pos_err_threshold (float): Position error threshold for matching.
            corr_err_threshold (float): Correlation threshold for matching.
        Returns:
            int: Maximum number of true positives.
        """
        # Extract positions and compute cost matrix for positions
        true_positions = np.array([atom['x'] for atom in true_atoms])
        approx_positions = np.array([atom['x'] for atom in approx_atoms])
        pos_cost_matrix = np.abs(true_positions[:, np.newaxis] - approx_positions)

        # Compute correlation matrix
        true_signals = []
        for atom in true_atoms:
            zs_atom = ZSAtom.from_dict(atom)
            zs_atom.padBothSides(self.dictionary.getAtomsLength())
            true_signals.append(zs_atom.getAtomInSignal(self.signals_length, atom['x']))
        approx_signals = []
        for atom in approx_atoms:
            zs_atom = ZSAtom.from_dict(atom)
            zs_atom.padBothSides(self.dictionary.getAtomsLength())
            approx_signals.append(zs_atom.getAtomInSignal(self.signals_length, atom['x']))
        correlation_matrix = np.zeros((len(true_atoms), len(approx_atoms)))

        for i, true_signal in enumerate(true_signals):
            for j, approx_signal in enumerate(approx_signals):
                correlation_matrix[i, j] = np.abs(np.correlate(true_signal, approx_signal, mode='valid')[0])

        # Define the cost matrix based on position and correlation thresholds
        cost_matrix = np.ones_like(pos_cost_matrix)
        cost_matrix[(pos_cost_matrix <= pos_err_threshold) & (correlation_matrix >= corr_err_threshold)] = 0

        # Compute the optimal matching
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
        max_tp = np.sum(cost_matrix[row_ind, col_ind] == 0)

        return max_tp

    def meanPositionError(self, true_atoms:List[Dict], approx_atoms:List[Dict]) -> List[int]:
        """
        Compute the position error between the true and approximation dictionaries.
        Args:
            true_atoms (List[Dict]): Atoms of the true dictionary.
            approx_atoms (List[Dict]): Atoms of the approximation dictionary.
        Returns:
            List[int]: List of position errors.
        """
        positions_errors = [true_atom['x'] - approx_atom['x'] for true_atom, approx_atom in self.computeMatchingPosition(true_atoms, approx_atoms)]
        return np.mean(positions_errors)
    
    def positionErrorPerStep(self, true_atoms:List[Dict], approx_atoms:List[Dict]) -> List[int]:
        """
        Compute the position error between the true and approximation dictionaries.
        Args:
            true_atoms (List[Dict]): Atoms of the true dictionary.
            approx_atoms (List[Dict]): Atoms of the approximation dictionary.
        Returns:
            List[int]: List of position errors.
        """
        positions_errors = [true_atom['x'] - approx_atom['x'] for true_atom, approx_atom in self.computeMatchingPosition(true_atoms, approx_atoms)]
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
    
    def computeOMPPositionErrorsPerStep(self, db_path:str) -> Dict:
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
            pos_error_per_step =self.positionErrorPerStep(signal_atoms, approx_atoms)
            # Append data
            for i, err in enumerate(pos_error_per_step):
                data_errors['id'].append(signal_id)
                data_errors['snr'].append(signal_dict['snr'])
                data_errors['sparsity'].append(signal_dict['sparsity'])
                data_errors['pos_err'].append(err)
                data_errors['abs_pos_err'].append(np.abs(err))
                data_errors['algo_step'].append(i+1)
        return data_errors
    
    def computeOMPPositionErrorsPerStep(self, db_path:str) -> Dict:
        """
        Compute the position errors for all the signal on a MMPDF database
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
            pos_error_per_step =self.positionErrorPerStep(signal_atoms, approx_atoms)
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
        data_errors = self.computeOMPPositionErrorsPerStep(db_path)
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
        data_errors = self.computeOMPPositionErrorsPerStep(db_path)
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
        data_errors = self.computeOMPPositionErrorsPerStep(db_path)
        df_all_steps = pd.DataFrame(data_errors)
        df = df_all_steps.loc[df_all_steps['algo_step'] == step]
        df = df.sort_values(by='abs_pos_err', ascending=ascending)
        return df
    
#                __    __     ______  
#               /\ "-./  \   /\  == \ 
#               \ \ \-./\ \  \ \  _-/ 
#                \ \_\ \ \_\  \ \_\   
#                 \/_/  \/_/   \/_/                                     
    
    def processMPResults(self, mp_db_path:str) -> pd.DataFrame:
        """
        Process the MMP-DF results.
        """
        results = []
        # Load the data
        data = self.loadDataFromPath(mp_db_path)
        # Iterate over the outputs
        for result in data['mp'] :
            # Get the MP result approximation
            signal_id = result['id']
            mp_approx_atoms = result['atoms']
            
            # Get the signal from the approx id
            signal_dict = self.signalDictFromId(signal_id)
            signal_atoms = signal_dict['atoms']
            snr = signal_dict['snr']

            result_mp = {
                'snr' : snr,
                'sparsity' : result['sparsity'],
                'algo_type' : 'MP',
                'true_atoms': signal_atoms,
                'predicted_atoms': mp_approx_atoms
            }

            results.append(result_mp)

        return {'results': results}


#            .         .                     .         .                          
#           ,8.       ,8.                   ,8.       ,8.          8 888888888o   
#          ,888.     ,888.                 ,888.     ,888.         8 8888    `88. 
#         .`8888.   .`8888.               .`8888.   .`8888.        8 8888     `88 
#        ,8.`8888. ,8.`8888.             ,8.`8888. ,8.`8888.       8 8888     ,88 
#       ,8'8.`8888,8^8.`8888.           ,8'8.`8888,8^8.`8888.      8 8888.   ,88' 
#      ,8' `8.`8888' `8.`8888.         ,8' `8.`8888' `8.`8888.     8 888888888P'  
#     ,8'   `8.`88'   `8.`8888.       ,8'   `8.`88'   `8.`8888.    8 8888         
#    ,8'     `8.`'     `8.`8888.     ,8'     `8.`'     `8.`8888.   8 8888         
#   ,8'       `8        `8.`8888.   ,8'       `8        `8.`8888.  8 8888         
#  ,8'         `         `8.`8888. ,8'         `         `8.`8888. 8 8888         
#  
    
    def plotMMPComparison(self, mmpdf_db_path:str, id:int) -> None :
        """
        Use three subplots to compare the results between the OMP and the MMP results.
        The OMP result corresponds to the first branch of the MMP tree.
        The MMP result corresponds to the MSE-argmin of the MMP tree.
        """
        # Load the data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)
            mmp_result_dict = next((result for result in output_data['mmp'] if result['id'] == id), None)
        
        # Get the true signal
        signal_dict = self.signalDictFromId(id)
        true_atoms = signal_dict['atoms']
        true_signal = np.zeros_like(signal_dict['signal'])
        for atom_dict in true_atoms :
            zs_atom = ZSAtom(atom_dict['b'], atom_dict['y'], atom_dict['s'])
            zs_atom.padBothSides(self.dictionary.getAtomsLength())
            atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom_dict['x'])
            true_signal += atom_signal

        fig, axs = plt.subplots(3, 1, figsize=(12, 3*3), sharex=True)
        axs[0].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
        
        # Find the OMP and the MMP dict
        min_mse = np.inf
        mmp_dict = None
        mmp_path = None
        mmp_tree_dict = mmp_result_dict['mmp-tree']

        for path_str, leaf_dict in mmp_tree_dict.items() :
            if all(c == '1' for c in path_str.split('-')) :
                omp_dict = leaf_dict
            if leaf_dict['mse'] <= min_mse :
                mmp_dict = leaf_dict
                min_mse = leaf_dict['mse']
                mmp_path = path_str

        # Extract the atoms from the dict
        results_dict = [omp_dict, mmp_dict]
        results_name = ['OMP', f'MMP {mmp_path}']

        # Plot the comparison
        for i, result_dict in enumerate(results_dict) :
            approx, _ = self.dictionary.getSignalProjectionFromAtoms(signal_dict['signal'], result_dict['atoms'])
            axs[0].plot(approx, color=f'C{i}', label=results_name[i])
            axs[i+1].plot(true_signal, color='g')
            axs[i+1].plot(approx, color=f'C{i}')
            axs[i+1].plot(signal_dict['signal'], color='k', alpha=0.4, lw=3)
            axs[i+1].set_title('{} : MSE = {}'.format(results_name[i], result_dict["mse"]), fontsize=12)
            axs[0].legend(loc='best')
            axs[i+1].axis('off')  
            
        axs[0].legend(loc='best') 
        axs[0].axis('off')
        plt.suptitle(f'OMP and MMP comparison on signal n°{id}', fontsize=14)
        plt.show()

    def plotMMPDecomposition(self, db_path:str, id:int, verbose:bool=True) -> None :
        """
        Plot the signal decomposition.
        Args:
            signal_dict (Dict): Signal dictionary.
        """
        # Load the data
        with open(db_path, 'r') as f:
            output_data = json.load(f)
            mmp_result_dict = next((result for result in output_data['mmp'] if result['id'] == id), None)
        # Get the true signal
        signal_dict = self.signalDictFromId(id)
        mmp_tree_dict = mmp_result_dict['mmp-tree']
        sparsity = mmp_result_dict['sparsity']

        # Find the OMP and the MMP dict
        min_mse = np.inf
        mmp_dict = None
        mmp_path = None
        for path_str, leaf_dict in mmp_tree_dict.items() :
            if all(c == '1' for c in path_str.split('-')) :
                omp_dict = leaf_dict
            if leaf_dict['mse'] <= min_mse :
                mmp_dict = leaf_dict
                mmp_path = path_str
                min_mse = leaf_dict['mse']

        # Extract the atoms from the dict
        omp_atoms_dict = omp_dict['atoms']
        mmp_atoms_dict = mmp_dict['atoms']

        if verbose :
            print(f'OMP atoms :')
            for atom in omp_atoms_dict :
                print(f'    {atom}')
            print(f'MMP-DF {mmp_path} atoms :')
            for atom in mmp_atoms_dict :
                print(f'    {atom}')

        nb_atoms = len(omp_atoms_dict)

        fig, axs = plt.subplots(nb_atoms+1, 1, figsize=(12, (nb_atoms+1)*3), sharex=True)
    
        # Create the OMP signal from the omp_atoms_dict
        omp_signal, omp_activations = self.dictionary.getSignalProjectionFromAtoms(signal_dict['signal'], omp_atoms_dict)
        omp_activations_indexes = np.argwhere(omp_activations > 0).flatten()

        # Create the OMP signal from the omp_atoms_dict
        mmp_signal, mmp_activations = self.dictionary.getSignalProjectionFromAtoms(signal_dict['signal'], mmp_atoms_dict)
        mmp_activations_indexes = np.argwhere(mmp_activations > 0).flatten()

        omp_atoms_signals = []
        mmp_atoms_signals = []

        for i, (omp_activation, mmp_activation) in enumerate(zip(omp_activations_indexes, mmp_activations_indexes)) :
            # Construct the atoms from parameters
            omp_atom, omp_offset = self.dictionary.getAtomFromActivationIdx(omp_activation)
            mmp_atom, mmp_offset = self.dictionary.getAtomFromActivationIdx(mmp_activation)

            # Get the atom signals
            omp_atom_signal = omp_atom.getAtomInSignal(len(signal_dict['signal']), omp_offset)
            mmp_atom_signal = mmp_atom.getAtomInSignal(len(signal_dict['signal']), omp_offset)
            
            # Plot the noisy signal
            axs[i+1].plot(signal_dict['signal'], label='Signal', color='k', alpha=0.3, lw=3)
            
            # Plot the current OMP atoms
            axs[i+1].plot(omp_atom_signal, label='OMP atom at step n°{i}', color='C0', alpha=1)
            for omp_atom_signal in omp_atoms_signals :
                axs[i+1].plot(omp_atom_signal, color='C0', alpha=0.3)
            omp_atoms_signals.append(omp_atom_signal)

            # Plot the current MMP atoms
            axs[i+1].plot(mmp_atom_signal, label='MMP atom at step n°{i}', color='C1', alpha=1)
            for mmp_atom_signal in mmp_atoms_signals :
                axs[i+1].plot(mmp_atom_signal, color='C1', alpha=0.3)
            mmp_atoms_signals.append(mmp_atom_signal)

            axs[i+1].set_title(f'Step n°{i+1}', fontsize=12)
            axs[i+1].legend(loc='best')
            axs[i+1].axis('off')

        omp_mse = np.mean((signal_dict['signal'] - omp_signal)**2)
        mmp_mse = np.mean((signal_dict['signal'] - mmp_signal)**2)

        if verbose :
            print(f'OMP MSE = {omp_mse}')
            print(f'MMP {mmp_path} MSE = {mmp_mse}')

        axs[0].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
        axs[0].plot(omp_signal, label='OMP')
        axs[0].plot(mmp_signal, label='MMP')
        axs[0].set_title('OMP MSE = {} & MMP {} MSE = {}'.format(omp_dict['mse'], mmp_path, mmp_dict['mse']), fontsize=12)
        axs[0].legend(loc='best')
        axs[0].axis('off')
        plt.show()

    def getArgminMSEFromMMPTree(self, mmp_tree_dict:Dict) -> Tuple[Dict, float]:
        """
        Get the argmin of the MSE from the MMP dictionary.
        """
        min_mse = np.inf
        min_leaf = None
        for path_str, leaf_dict in mmp_tree_dict.items() :
            if leaf_dict['mse'] <= min_mse :
                min_mse = leaf_dict['mse']
                min_leaf = leaf_dict
        return min_leaf, min_mse
    
    def computeMMPPositionErrorsPerStep(self, db_path:str) -> Dict:
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
            'algo_step': [],
            'algo_type': []
        }
        # Iterate over the outputs
        for result in output_data['mmp'] :

            # Get the MMP approximation
            signal_id = result['id']
            mmp_tree_dict = result['mmp-tree']
            mmp_approx_dict, mmp_approx_mse = self.getArgminMSEFromMMPTree(mmp_tree_dict)
            mmp_approx_atoms = mmp_approx_dict['atoms']

            # Get the OMP approximation
            omp_path_str = '-'.join(['1']*result['sparsity'])
            omp_approx_mse = mmp_tree_dict[omp_path_str]['mse']
            omp_approx_atoms = mmp_tree_dict[omp_path_str]['atoms']

            # Get the signal from the approx id
            signal_dict = self.signalDictFromId(signal_id)
            signal_atoms = signal_dict['atoms']

            # Compute MMP errors
            mmp_pos_error_per_step = self.positionErrorPerStep(signal_atoms, mmp_approx_atoms)
            # Append MMP data
            for i, err in enumerate(mmp_pos_error_per_step):
                data_errors['id'].append(signal_id)
                data_errors['snr'].append(signal_dict['snr'])
                data_errors['sparsity'].append(signal_dict['sparsity'])
                data_errors['pos_err'].append(err)
                data_errors['abs_pos_err'].append(np.abs(err))
                data_errors['algo_step'].append(i+1)
                data_errors['algo_type'].append('MMP-DF')

            # Compute OMP errors
            omp_pos_error_per_step = self.positionErrorPerStep(signal_atoms, omp_approx_atoms)
            # Append OMP data
            for i, err in enumerate(omp_pos_error_per_step):
                data_errors['id'].append(signal_id)
                data_errors['snr'].append(signal_dict['snr'])
                data_errors['sparsity'].append(signal_dict['sparsity'])
                data_errors['pos_err'].append(err)
                data_errors['abs_pos_err'].append(np.abs(err))
                data_errors['algo_step'].append(i+1)
                data_errors['algo_type'].append('OMP')
        return data_errors
    
    def sortByOMPPositionErrorAtStep(self, mmpdf_db_path:str, step:int, ascending:bool=True) :
        """
        Sort the position errors by the OMP position error at a given step.
        """
        data_errors = self.computeMMPPositionErrorsPerStep(mmpdf_db_path)
        df_all_type = pd.DataFrame(data_errors)
        df_omp = df_all_type.loc[df_all_type['algo_type'] == 'OMP']
        df = df_omp.loc[df_omp['algo_step'] == step]
        df = df.sort_values(by='abs_pos_err', ascending=ascending)
        return df
    
    def computeMMPMSE(self, mmpdf_db_path:str) -> Dict:
        """
        Compute the position errors for all the signals.
        """
        # Load the data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)

        data_errors = {
            'id': [],
            'snr': [],
            'sparsity': [],
            'omp-mse': [],  
            'mmp-mse': [],
            'mse-diff': [],
        }
        # Iterate over the outputs
        for result in output_data['mmp'] :

            # Get the MMP approximation
            signal_id = result['id']
            mmp_tree_dict = result['mmp-tree']
            mmp_approx_dict, mmp_approx_mse = self.getArgminMSEFromMMPTree(mmp_tree_dict)
            mmp_approx_atoms = mmp_approx_dict['atoms']

            # Get the OMP approximation
            omp_path_str = '-'.join(['1']*result['sparsity'])
            omp_approx_mse = mmp_tree_dict[omp_path_str]['mse']
            omp_approx_atoms = mmp_tree_dict[omp_path_str]['atoms']

            # Get the signal from the approx id
            signal_dict = self.signalDictFromId(signal_id)
            signal_atoms = signal_dict['atoms']

            # Compute errors
            data_errors['id'].append(signal_id)
            data_errors['snr'].append(signal_dict['snr'])
            data_errors['sparsity'].append(signal_dict['sparsity'])
            data_errors['omp-mse'].append(omp_approx_mse)
            data_errors['mmp-mse'].append(mmp_approx_mse)
            data_errors['mse-diff'].append((omp_approx_mse - mmp_approx_mse)/omp_approx_mse)
            
        return data_errors
    
    def sortByOMPMSE(self, mmpdf_db_path:str, ascending:bool=True) :
        """
        Sort the position errors by the OMP position error at a given step.
        """
        data_errors = self.computeMMPMSE(mmpdf_db_path)
        df= pd.DataFrame(data_errors)
        df = df.sort_values(by='omp-mse', ascending=ascending)
        return df

    def sortByBestMSEDiff(self, mmpdf_db_path:str, ascending:bool=True) :
        """
        Sort the position errors by the relative difference between OMP and MMP MSE
        """
        data_errors = self.computeMMPMSE(mmpdf_db_path)
        df= pd.DataFrame(data_errors)
        df = df.sort_values(by='mse-diff', ascending=ascending)
        return df
        

    def boxplotMMPDFPosErrComparison(self, mmpdf_db_path:str) :
        """
        Plot the boxplot of the position errors
        Args:
            mmpdf_db_path (str): Path to the MMPDF database.
        """
        plt.figure(figsize=(12, 8)) 
        data_errors = self.computeMMPPositionErrorsPerStep(mmpdf_db_path)
        df = pd.DataFrame(data_errors)
        sns.boxplot(x='snr', y='pos_err', hue='algo_type', data=df, palette="flare", fliersize=2, whis=1.5, showfliers=False)
        sns.despine(trim=True)
        plt.title('OMP vs MMPDF Position Errors Comparison by SNR', fontsize=14)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=12)
        plt.ylabel('Position Error', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Algo', loc='best')
        plt.show()

    def boxplotMMPDFPosErrAtSparsity(self, mmpdf_db_path:str, sparsity:int) :
        """
        Plot the boxplot of the position errors for a given sparsity.
        Args:
            mmpdf_db_path (str): Path to the MMPDF database.
            sparsity (int): Sparsity level.
        """
        plt.figure(figsize=(12, 8)) 
        data_errors = self.computeMMPPositionErrorsPerStep(mmpdf_db_path)
        df_all_sparsities = pd.DataFrame(data_errors)
        df = df_all_sparsities.loc[df_all_sparsities['sparsity'] == sparsity]
        sns.boxplot(x='snr', y='pos_err', hue='algo_type', data=df, palette="flare", fliersize=2, whis=1.5, showfliers=False)
        sns.despine(trim=True)
        plt.title(f'OMP vs MMPDF Position Errors Comparison by SNR at sparsity={sparsity}', fontsize=14)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=12)
        plt.ylabel('Position Error', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Algo', loc='best')
        plt.show()

    def boxplotMMPDFPosErrAtStep(self, mmpdf_db_path:str, step:int) :
        """
        Plot the boxplot of the position errors for a given sparsity.
        Args:
            mmpdf_db_path (str): Path to the MMPDF database.
            sparsity (int): Sparsity level.
        """
        plt.figure(figsize=(12, 8)) 
        data_errors = self.computeMMPPositionErrorsPerStep(mmpdf_db_path)
        df_all_steps = pd.DataFrame(data_errors)
        df = df_all_steps.loc[df_all_steps['algo_step'] == step]
        sns.boxplot(x='snr', y='pos_err', hue='algo_type', data=df, palette="flare", fliersize=2, whis=1.5, showfliers=False)
        sns.despine(trim=True)
        plt.title(f'OMP vs MMPDF Position Errors Comparison by SNR at step={step}', fontsize=14)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=12)
        plt.ylabel('Position Error', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Algo', loc='best')
        plt.show()

    def processMMPDFResults(self, db_path:str) -> pd.DataFrame:
        """
        Process the MMP-DF results.
        """
        results = []
        # Load the data
        data = self.loadDataFromPath(db_path)
        # Iterate over the outputs
        for result in data['mmp'] :
            # Get the MMP approximation
            signal_id = result['id']
            mmp_tree_dict = result['mmp-tree']
            mmp_approx_dict, mmp_approx_mse = self.getArgminMSEFromMMPTree(mmp_tree_dict)
            mmp_approx_atoms = mmp_approx_dict['atoms']

            # Get the OMP approximation
            omp_path_str = '-'.join(['1']*result['sparsity'])
            omp_approx_mse = mmp_tree_dict[omp_path_str]['mse']
            omp_approx_atoms = mmp_tree_dict[omp_path_str]['atoms']

            # Get the signal from the approx id
            signal_dict = self.signalDictFromId(signal_id)
            signal_atoms = signal_dict['atoms']
            snr = signal_dict['snr']

            result_omp = {
                'snr' : snr,
                'sparsity' : signal_dict['sparsity'],
                'algo_type' : 'OMP',
                'true_atoms': signal_atoms,
                'predicted_atoms': omp_approx_atoms
            }

            result_mmp = {
                'snr' : snr,
                'sparsity' : signal_dict['sparsity'],
                'algo_type' : 'MMP-DF',
                'true_atoms': signal_atoms,
                'predicted_atoms': mmp_approx_atoms
            }

            results.append(result_omp)
            results.append(result_mmp)

        return {'results': results}
    
    def computeScoreF1Position(self, matched_atoms):
        """
        Compute the F1 score based on the position errors.
        Returns:
            tp (int): True positives.
            fp (int): False positives.
            fn (int): False negatives.
        Args:
            matched_atoms (List): List of matched atoms.
        """
        position_errors = [abs(true_atom['x'] - predicted_atom['x']) for true_atom, predicted_atom in matched_atoms]
        tp = sum(1 for error in position_errors if error <= 5)
        fp = len(matched_atoms) - tp
        fn = len(matched_atoms) - tp
        return tp, fp, fn
    
    def computeScoreF1Correlation(self, matched_atoms):
        """
        Compute the F1 score based on the correlation errors.
        Returns:
            tp (int): True positives.
            fp (int): False positives.
            fn (int): False negatives.
        Args:
            matched_atoms (List): List of matched atoms.
        """
        corr_errors = []
        for true_atom, predicted_atom in matched_atoms:
            # Build the ture atom's signal
            true_atom_obj = ZSAtom.from_dict(true_atom)
            true_atom_obj.padBothSides(self.dictionary.getAtomsLength())
            true_atom_signal = true_atom_obj.getAtomInSignal(self.signals_length, true_atom['x'])
            # Build the predicted atom's signal 
            predicted_atom_obj = ZSAtom.from_dict(predicted_atom)
            predicted_atom_obj.padBothSides(self.dictionary.getAtomsLength())
            predicted_atom_signal = predicted_atom_obj.getAtomInSignal(self.signals_length, predicted_atom['x'])
            corr = np.correlate(true_atom_signal, predicted_atom_signal, mode='valid')[0]
            corr_errors.append(corr)
        tp = sum(1 for corr in corr_errors if abs(corr) >= self.f1CorrThreshold)
        fp = len(matched_atoms) - tp
        fn = len(matched_atoms) - tp
        return tp, fp, fn

    def calculateF1Metrics(self, row:pd.Series, matching:str, f1:str) -> pd.Series:
        """
        Calculate the F1 score.
        """
        true_atoms = row['true_atoms']
        predicted_atoms = row['predicted_atoms']

        # Atom matching
        matching_method = getattr(self, f'computeMatching{matching.capitalize()}')
        matched_atoms = matching_method(true_atoms, predicted_atoms)

        # F1 metrics
        metrics_method = getattr(self, f'computeScoreF1{f1.capitalize()}')
        tp, fp, fn = metrics_method(matched_atoms)

        return pd.Series([tp, fp, fn])
    
    def metrics_positionMatching_positionF1(self, row) :
        return self.calculateF1Metrics(row, matching='position', f1='position')

    def metrics_positionMatching_correlationF1(self, row) :
        return self.calculateF1Metrics(row, matching='position', f1='correlation')
    
    def metrics_correlationMatching_positionF1(self, row) :
        return self.calculateF1Metrics(row, matching='correlation', f1='position')
        
    def metrics_correlationMatching_correlationF1(self, row) :
        return self.calculateF1Metrics(row, matching='correlation', f1='correlation')

    def computeMMPDFScoreF1(self, db_path, matching_type='position', f1_type='position'):
        """
        Compute the F1 score by SNR for different algorithms and sparsity levels.
        """
        processed_results = self.processMMPDFResults(db_path)
        # Convert data to DataFrame
        df = pd.DataFrame(processed_results['results'])

        # Get the metrics function
        metrics_func = getattr(self, f'metrics_{matching_type}Matching_{f1_type}F1')
        metrics = df.apply(metrics_func, axis=1)
        df[['tp', 'fp', 'fn']] = metrics
        
        # Calculate Precision, Recall, F1 Score
        df['precision'] = df['tp'] / (df['tp'] + df['fp'])
        df['recall'] = df['tp'] / (df['tp'] + df['fn'])
        df['F1'] = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])
    
        return df

    def plotMMPDFScoreF1(self, db_path: str, matching_type: str = 'position', f1_type: str = 'position'):
        """
        Plot the F1 score by S(NR for different algorithms and sparsity levels.
        """
        plt.figure(figsize=(12, 8))

        metrics_df = self.computeMMPDFScoreF1(db_path, matching_type, f1_type)

        # Define colors and markers for the plots
        colors = {'OMP': 'navy', 'MMP-DF': 'red'}
        markers = {3: 'o', 4: 'D', 5: 'X'}  # Example for up to 5 sparsity level

        # Group data and plot
        grouped = metrics_df.groupby(['algo_type', 'sparsity'])
        for (algo_type, sparsity), group in grouped:
            sns.lineplot(x='snr', y='F1', data=group,
                        label=f'{algo_type} Sparsity {sparsity}',
                        color=colors[algo_type],
                        marker=markers[sparsity],
                        markersize=8)

        plt.title(f'OMP and MMP-DF F1 Scores by SNR : matching:{matching_type}, f1:{f1_type}', fontsize=16)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Algorithm & Sparsity', loc='upper left', fontsize=12)
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        plt.show()

    def plotMMPDFScoreF1Global(self, db_path: str, matching_type: str = 'position', f1_type: str = 'position'):
        """
        Plot the F1 score by SNR for different algorithms and sparsity levels.
        """
        plt.figure(figsize=(12, 8))

        metrics_df = self.computeMMPDFScoreF1(db_path, matching_type, f1_type)

        # Define colors and markers for the plots
        colors = {'OMP': 'navy', 'MMP-DF': 'red'}
        markers = {3: 'o', 4: 'D', 5: 'X'}  # Example for up to 5 sparsity levels

        # Group data and plot
        grouped = metrics_df.groupby(['algo_type'])
        i=3
        for (algo_type,), group in grouped:
            sns.lineplot(x='snr', y='F1', data=group,
                        label=f'{algo_type}',
                        color=colors[algo_type],
                        marker=markers[i],
                        markersize=8)
            i+=1

        plt.title(f'OMP and MMP-DF F1 Scores by SNR : matching:{matching_type}, f1:{f1_type}', fontsize=16)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Algorithm & Sparsity', loc='upper left', fontsize=12)
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        plt.show()
    
 #       ______     __   __   ______     ______     __         ______     ______  
 #      /\  __ \   /\ \ / /  /\  ___\   /\  == \   /\ \       /\  __ \   /\  == \ 
 #      \ \ \/\ \  \ \ \'/   \ \  __\   \ \  __<   \ \ \____  \ \  __ \  \ \  _-/ 
 #       \ \_____\  \ \__|    \ \_____\  \ \_\ \_\  \ \_____\  \ \_\ \_\  \ \_\   
 #        \/_____/   \/_/      \/_____/   \/_/ /_/   \/_____/   \/_/\/_/   \/_/   
 #                                                                                
    def getSignalOverlapVectorFromId(self, id:int) -> np.ndarray:
        """
        Get the signal overlap vector from its ID.
        It is a integer vector that counts the number of atoms that overlap at each position.
        """
        # Get the signal dictionary
        signal_dict = self.signalDictFromId(id)
        atoms_list = signal_dict['atoms']
        atoms_signals = list()
        # Compute the atoms signals
        for atom in atoms_list:
            zs_atom = ZSAtom.from_dict(atom)
            zs_atom.padBothSides(self.dictionary.getAtomsLength())
            atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom['x'])
            atoms_signals.append(atom_signal)
        # Build the binary support vector for each atom
        bin_atoms_signals = [np.where(np.abs(atom_signal) > ZSAtom.SUPPORT_THRESHOLD, 1, 0) for atom_signal in atoms_signals]
        overlap_vector = sum(bin_atoms_signals)
        return overlap_vector
    
    def getSignalOverlapIntervalsFromId(self, id:int) -> List:
        """
        Get the signal overlap intervals from its ID.
        It is a list of tuples that contains the start and end positions of the overlap intervals.
        Returns:
            overlap_intervals (List): List of tuples (start, end)
            overlap_intervals_values (List): List of overlap values for each interval.
        """
        overlap_vector = self.getSignalOverlapVectorFromId(id)
        overlap_intervals = list()
        overlap_intervals_values = list()
        start = 0
        current_overlap = overlap_vector[0]
        for idx, overlap in enumerate(overlap_vector):
            if current_overlap != overlap:
                overlap_intervals.append((start, idx))
                overlap_intervals_values.append(current_overlap)
                current_overlap = overlap
                start = idx
        overlap_intervals.append((start, len(overlap_vector)))
        overlap_intervals_values.append(current_overlap)   
        return overlap_intervals, overlap_intervals_values

    def getSignalOverlapMetricsFromId(self, id:int) -> Dict:
        """
        Get the signal overlap metrics from its ID.
        It is a dictionary that counts the number of positions that are overlapped by a given number of atoms.
        Key = number of atoms that overlap a position : Value = number of positions.
        Args:
            id (int): Signal ID.
        Returns:
            dict: Signal dictionary.
        """
        signal_dict = self.signalDictFromId(id)
        atoms_list = signal_dict['atoms']
        overlap_vector = self.getSignalOverlapVectorFromId(id)
        overlap_metrics = dict()
        for overlap in range(len(atoms_list)+1) :
            overlap_metrics[overlap] = sum(overlap_vector >= overlap)
        return overlap_metrics

    def signalOverlapPrctFromId(self, id:int, nb_round:int=2) -> Dict:
        """
        Get the signal overlap percentage from its ID.
        """
        overlap_metrics = self.getSignalOverlapMetricsFromId(id)
        total_steps = sum(overlap_metrics.values())
        overlap_prct = {overlap: round(100*count/total_steps, nb_round) for overlap, count in overlap_metrics.items()}
        return overlap_prct
    
    def signalOverlapTypePrctFromId(self, id:int, nb_round:int=2) -> Dict:
        """
        Get the signal overlap type percentage from its ID.
        """
        overlap_metrics = self.getSignalOverlapMetricsFromId(id)
        total_steps = sum(overlap_metrics.values())
        prct_overlap_type = dict()
        for overlap_level in range(1, len(overlap_metrics)) :
            counter = sum([count for overlap, count in overlap_metrics.items() if overlap >= overlap_level])
            prct = round(100*counter/total_steps, nb_round)
            key = f'>={overlap_level}'
            prct_overlap_type[key] = prct
        return prct_overlap_type

    def plotSignalOverlapFromId(self, id:int) -> None :
        """
        Plot the signal with a conditional coloring according to the overlap vector.
        """
        # Get the signal dictionary
        signal_dict = self.signalDictFromId(id)
        atoms_list = signal_dict['atoms']
        overlap_vector = self.getSignalOverlapVectorFromId(id)

        # Plot the signal
        fig, axs = plt.subplots(2, 1, figsize=(12, 2*2), sharex=True)
        axs[0].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
        true_signal = np.zeros_like(signal_dict['signal'])

        # Color the background based on overlap
        cmap = plt.get_cmap('plasma')
        max_overlap = max(overlap_vector)
        norm = Normalize(vmin=0, vmax=max_overlap)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for idx, val in enumerate(overlap_vector):
            axs[0].axvspan(idx, idx+1, facecolor=cmap(norm(val)), alpha=0.3)
            axs[1].axvspan(idx, idx+1, facecolor=cmap(norm(val)), alpha=0.3)

        # Compute the atoms signals
        offset = 1.5*max(np.abs(signal_dict['signal']))
        for i, atom in enumerate(atoms_list):
            zs_atom = ZSAtom.from_dict(atom)
            zs_atom.padBothSides(self.dictionary.getAtomsLength())
            atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom['x'])
            true_signal += atom_signal
            # Plot the atom's signal
            axs[1].plot(atom_signal + i*offset, label=f'Atom at {atom["x"]}', alpha=0.6, lw=2)

        axs[0].plot(true_signal, label='True signal', color='g', lw=2)   
        axs[0].legend(loc='best')
        axs[0].axis('off')
        #axs[1].legend(loc='best')
        axs[1].axis('off')

        # Add a colorbar
        fig.colorbar(sm, ax=axs, orientation='vertical', label='Overlap level', pad=0.01)
        plt.show()

    def plotSignalOverlapErrorFromId(self, mmpdf_db_path:str, id:int) -> None :
        """
        Plot the signal with a conditional coloring according to the overlap vector.
        """
        # Load the data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)
            mmp_result_dict = next((result for result in output_data['mmp'] if result['id'] == id), None)
        
        # Get the true signal
        signal_dict = self.signalDictFromId(id)
        mmp_tree_dict = mmp_result_dict['mmp-tree']

        # Find the OMP and the MMP dict
        min_mse = np.inf
        mmp_dict = None
        mmp_path = None
        for path_str, leaf_dict in mmp_tree_dict.items() :
            if all(c == '1' for c in path_str.split('-')) :
                omp_dict = leaf_dict
            if leaf_dict['mse'] <= min_mse :
                mmp_dict = leaf_dict
                mmp_path = path_str
                min_mse = leaf_dict['mse']

        # Extract the atoms from the dict
        omp_atoms_dict = omp_dict['atoms']
        mmp_atoms_dict = mmp_dict['atoms']

        # Create the signals
        omp_signal = np.zeros_like(signal_dict['signal'])
        mmp_signal = np.zeros_like(signal_dict['signal'])

        for i, (omp_atom, mmp_atom) in enumerate(zip(omp_atoms_dict, mmp_atoms_dict)) :
            # Construct the atoms from parameters
            omp_zs_atom = ZSAtom(omp_atom['b'], omp_atom['y'], omp_atom['s'])
            omp_zs_atom.padBothSides(self.dictionary.getAtomsLength())
            mmp_zs_atom = ZSAtom(mmp_atom['b'], mmp_atom['y'], mmp_atom['s'])
            mmp_zs_atom.padBothSides(self.dictionary.getAtomsLength())
            # Get the atom signals
            omp_atom_signal = omp_zs_atom.getAtomInSignal(len(signal_dict['signal']), omp_atom['x'])
            omp_signal += omp_atom_signal
            mmp_atom_signal = mmp_zs_atom.getAtomInSignal(len(signal_dict['signal']), mmp_atom['x'])
            mmp_signal += mmp_atom_signal

        # Get the overlap vector of the signal
        overlap_vector = self.getSignalOverlapVectorFromId(id)

        # Build the figure
        fig, axs = plt.subplots(2, 1, figsize=(15, 3*2), sharex=True)
        
        # Color the background based on overlap
        cmap = plt.get_cmap('plasma')
        max_overlap = max(overlap_vector)
        norm = Normalize(vmin=0, vmax=max_overlap)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for idx, val in enumerate(overlap_vector):
            axs[0].axvspan(idx, idx+1, facecolor=cmap(norm(val)), alpha=0.3)
            axs[1].axvspan(idx, idx+1, facecolor=cmap(norm(val)), alpha=0.3)

        # Plot the signals
        axs[0].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
        axs[0].plot(omp_signal, label='OMP', color='b', lw=2)
        axs[0].plot(mmp_signal, label=f'MMP {mmp_path}', color='r', lw=2)
        axs[0].legend(loc='best')
        axs[0].axis('off')

        # Compute the reconstruction error
        omp_quadratic_error = (signal_dict['signal'] - omp_signal)**2
        mmp_quadratic_error = (signal_dict['signal'] - mmp_signal)**2

        # Plot the reconstruction error
        axs[1].plot(omp_quadratic_error, label='OMP error', color='b', lw=1, alpha=0.9)
        axs[1].plot(mmp_quadratic_error, label=f'MMP {mmp_path} error', color='r', lw=1, alpha=0.9)
        axs[1].legend(title='Quadratic error', loc='best')
        axs[1].axis('off')

        # Add a colorbar
        fig.colorbar(sm, ax=axs, orientation='vertical', label='Overlap level', pad=0.01)
        plt.show()

    def plotSignalOverlapIntervalMSEFromId(self, mmpdf_db_path:str, id:int) -> None :
        """
        Plot the signal with a conditional coloring according to the overlap vector.
        """
        # Load the data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)
            mmp_result_dict = next((result for result in output_data['mmp'] if result['id'] == id), None)
        
        # Reconstruct the denoised signal
        signal_dict = self.signalDictFromId(id)
        signal_atoms = signal_dict['atoms']
        true_signal = np.zeros_like(signal_dict['signal'])
        for atom in signal_atoms:
            zs_atom = ZSAtom.from_dict(atom)
            zs_atom.padBothSides(self.dictionary.getAtomsLength())
            atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom['x'])
            true_signal += atom_signal     

        # Find the OMP and the MMP dict
        mmp_tree_dict = mmp_result_dict['mmp-tree']
        min_mse = np.inf
        mmp_dict = None
        mmp_path = None
        for path_str, leaf_dict in mmp_tree_dict.items() :
            if all(c == '1' for c in path_str.split('-')) :
                omp_dict = leaf_dict
            if leaf_dict['mse'] <= min_mse :
                mmp_dict = leaf_dict
                mmp_path = path_str
                min_mse = leaf_dict['mse']

        # Extract the atoms from the dict
        omp_atoms_dict = omp_dict['atoms']
        mmp_atoms_dict = mmp_dict['atoms']

        # Create the signals
        omp_signal = np.zeros_like(signal_dict['signal'])
        mmp_signal = np.zeros_like(signal_dict['signal'])

        for i, (omp_atom, mmp_atom) in enumerate(zip(omp_atoms_dict, mmp_atoms_dict)) :
            # Construct the atoms from parameters
            omp_zs_atom = ZSAtom(omp_atom['b'], omp_atom['y'], omp_atom['s'])
            omp_zs_atom.padBothSides(self.dictionary.getAtomsLength())
            mmp_zs_atom = ZSAtom(mmp_atom['b'], mmp_atom['y'], mmp_atom['s'])
            mmp_zs_atom.padBothSides(self.dictionary.getAtomsLength())
            # Get the atom signals
            omp_atom_signal = omp_zs_atom.getAtomInSignal(len(signal_dict['signal']), omp_atom['x'])
            omp_signal += omp_atom_signal
            mmp_atom_signal = mmp_zs_atom.getAtomInSignal(len(signal_dict['signal']), mmp_atom['x'])
            mmp_signal += mmp_atom_signal

        # Build the figure
        fig, axs = plt.subplots(2, 1, figsize=(15, 3*2), sharex=True)
        
        # Get the overlap vector of the signal
        overlap_vector = self.getSignalOverlapVectorFromId(id)
        overlap_intervals, overlap_intervals_values = self.getSignalOverlapIntervalsFromId(id)

        # Color the background based on overlap
        cmap = plt.get_cmap('plasma')
        max_overlap = max(overlap_vector)
        norm = Normalize(vmin=0, vmax=max_overlap)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for idx, val in enumerate(overlap_vector):
            axs[0].axvspan(idx, idx+1, facecolor=cmap(norm(val)), alpha=0.3)
            axs[1].axvspan(idx, idx+1, facecolor=cmap(norm(val)), alpha=0.3)

        # Plot the signals
        axs[0].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
        axs[0].plot(omp_signal, label='OMP', color='b', lw=2)
        axs[0].plot(mmp_signal, label=f'MMP {mmp_path}', color='r', lw=2)
        axs[0].legend(loc='best')
        axs[0].axis('off')

        omp_mse_signal = np.zeros_like(true_signal)
        mmp_mse_signal = np.zeros_like(true_signal)
        for (start, end), val in zip(overlap_intervals, overlap_intervals_values):
            # Compute the mse on the interval
            omp_mse_on_interval = np.mean((true_signal[start:end] - omp_signal[start:end])**2)
            mmp_mse_on_interval = np.mean((true_signal[start:end] - mmp_signal[start:end])**2)
            # Fill the mse signals
            omp_mse_signal[start:end] = omp_mse_on_interval*np.ones(end-start)
            mmp_mse_signal[start:end] = mmp_mse_on_interval*np.ones(end-start)

        # Plot the constant by interval MSE
        axs[1].plot(omp_mse_signal, label='OMP error', color='b', lw=1, alpha=0.9)
        axs[1].plot(mmp_mse_signal, label=f'MMP {mmp_path} error', color='r', lw=1, alpha=0.9)
        axs[1].legend(title='MSE per interval', loc='best')
        axs[1].axis('off')

        # Add a colorbar
        fig.colorbar(sm, ax=axs, orientation='vertical', label='Overlap level', pad=0.01)
        plt.show()

    def computeMMPDFLocalMSEPerOverlapInterval(self, db_path:str) -> Dict:
        """
        Compute the MSE per overlap interval.
        Each row corresponds to an overlap interval.
        """
        # Load the data
        with open(db_path, 'r') as f:
            output_data = json.load(f)

        data_overlap_intervals = {
            'id': [],
            'snr': [],
            'overlap': [],
            'local_mse': [],
            'algo_type': []
        }
        # Iterate over the outputs
        for result in output_data['mmp'] :

            # Reconstruct the denoised signal
            signal_id = result['id']
            signal_dict = self.signalDictFromId(signal_id)
            signal_atoms = signal_dict['atoms']
            true_signal = np.zeros_like(signal_dict['signal'])
            for atom in signal_atoms:

                zs_atom = ZSAtom.from_dict(atom)
                zs_atom.padBothSides(self.dictionary.getAtomsLength())
                atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom['x'])
                true_signal += atom_signal            

            # Get the MMP approximation
            mmp_tree_dict = result['mmp-tree']
            mmp_approx_dict, mmp_approx_mse = self.getArgminMSEFromMMPTree(mmp_tree_dict)
            mmp_approx_atoms = mmp_approx_dict['atoms']

            # Get the OMP approximation
            omp_path_str = '-'.join(['1']*result['sparsity'])
            omp_approx_mse = mmp_tree_dict[omp_path_str]['mse']
            omp_approx_atoms = mmp_tree_dict[omp_path_str]['atoms']

            # Get the overlap vector of the signal
            overlap_vector = self.getSignalOverlapVectorFromId(signal_id)
            overlap_intervals, overlap_intervals_values = self.getSignalOverlapIntervalsFromId(signal_id)

            # Compute the reconstruction signals
            # Create the signals
            omp_signal = np.zeros_like(signal_dict['signal'])
            mmp_signal = np.zeros_like(signal_dict['signal'])

            for i, (omp_atom, mmp_atom) in enumerate(zip(omp_approx_atoms, mmp_approx_atoms)) :
                # Construct the atoms from parameters
                omp_zs_atom = ZSAtom(omp_atom['b'], omp_atom['y'], omp_atom['s'])
                omp_zs_atom.padBothSides(self.dictionary.getAtomsLength())
                mmp_zs_atom = ZSAtom(mmp_atom['b'], mmp_atom['y'], mmp_atom['s'])
                mmp_zs_atom.padBothSides(self.dictionary.getAtomsLength())
                # Get the atom signals
                omp_atom_signal = omp_zs_atom.getAtomInSignal(len(signal_dict['signal']), omp_atom['x'])
                omp_signal += omp_atom_signal
                mmp_atom_signal = mmp_zs_atom.getAtomInSignal(len(signal_dict['signal']), mmp_atom['x'])
                mmp_signal += mmp_atom_signal

            # Compute the local reconstruction error for each overlap interval
            for (start, end), overlap in zip(overlap_intervals, overlap_intervals_values):
                # Compute the mse on the interval
                omp_mse_on_interval = np.mean((true_signal[start:end] - omp_signal[start:end])**2)
                mmp_mse_on_interval = np.mean((true_signal[start:end] - mmp_signal[start:end])**2)

                # Append the OMP data 
                data_overlap_intervals['id'].append(signal_id)
                data_overlap_intervals['snr'].append(signal_dict['snr'])
                data_overlap_intervals['overlap'].append(overlap)
                data_overlap_intervals['local_mse'].append(omp_mse_on_interval)
                data_overlap_intervals['algo_type'].append('OMP')

                # Append the MMP data
                data_overlap_intervals['id'].append(signal_id)
                data_overlap_intervals['snr'].append(signal_dict['snr'])
                data_overlap_intervals['overlap'].append(overlap)
                data_overlap_intervals['local_mse'].append(mmp_mse_on_interval)
                data_overlap_intervals['algo_type'].append('MMP-DF')
            
        return data_overlap_intervals

    def plotMMPDFLocalMSEOverlapBoxplot(self, mmpdf_db_path:str, snr:int=-5) :
        """
        Plot the boxplot of the position errors
        Args:
            mmpdf_db_path (str): Path to the MMPDF database.
        """
        plt.figure(figsize=(12, 8)) 
        data_overlap_intervals = self.computeMMPDFLocalMSEPerOverlapInterval(mmpdf_db_path)
        df_all_snr = pd.DataFrame(data_overlap_intervals)
        df = df_all_snr.loc[df_all_snr['snr'] == snr]
        sns.boxplot(x='overlap', y='local_mse', hue='algo_type', data=df, palette="flare", fliersize=2, whis=1.5, showfliers=False)
        sns.despine(trim=True)
        plt.title('OMP vs MMPDF local MSE Comparison by local overlap level', fontsize=14)
        plt.xlabel('Overlap >=', fontsize=12)
        plt.ylabel('Local MSE', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Algorithm', loc='best')
        plt.show()

    def computeMSEPerOverlapInterval_MMPDF(self, mmpdf_sparVar_db_path:str) -> Dict:
        pass

    def computeMSEPerOverlapInterval_OMP(self, db_path:str) -> Dict:
        pass

    def computeMSEPerOverlapInterval_MP(self, db_path:str) -> Dict:
        pass

    def plotMSEOverlapBoxplot(self, snr:int=-5, **kwargs) :
        """
        Plot the boxplot of the position errors
        Args:
            mmpdf_db_path (str): Path to the MMPDF database.
        """
        plt.figure(figsize=(12, 8)) 

        # Compute the local MSE per overlap interval for each algorithm
        algo_intervals = dict()
        for key, value in kwargs.items():
            if 'MMP' in key.lower() :
                algo_intervals[key] = self.computeMSEPerOverlapInterval_MMPDF(value)
            elif 'OMP' in key.lower() :
                algo_intervals[key] = self.computeMSEPerOverlapInterval_OMP(value)
            elif 'MP' in key.lower() :
                algo_intervals[key] = self.computeMSEPerOverlapInterval_MP(value)
            else :
                raise ValueError(f'Unknown algorithm type : {key}')

        # Plot the boxplot for each algorithm
        for algorithm, dict_intervals in algo_intervals.items():
            df_all_snr = pd.DataFrame(dict_intervals)
            df = df_all_snr.loc[df_all_snr['snr'] == snr]
            sns.boxplot(x='overlap', y='local_mse', hue='algo_type', data=df, palette="flare", fliersize=2, whis=1.5, showfliers=False)
            
        sns.despine(trim=True)
        plt.title('OMP vs MMPDF local MSE Comparison by local overlap level', fontsize=14)
        plt.xlabel('Overlap >=', fontsize=12)
        plt.ylabel('Local MSE', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Algorithm', loc='best')
        plt.show()

    def computeMMPDFLocalMSEPOverlapPrct(self, db_path:str) -> Dict:
        """
        Compute the MSE and the overlap percentage for each signal.
        Each row corresponds to a signal.
        """
        # Load the data
        with open(db_path, 'r') as f:
            output_data = json.load(f)

        data_prct_overlap_type = {
            'id': [],
            'snr': [],
            'overlap_type': [],
            'prct_in_signal': [],
            'algo_type': [],
            'mse': []
        }
        # Iterate over the outputs
        for result in output_data['mmp'] :

            # Reconstruct the denoised signal
            signal_id = result['id']
            signal_dict = self.signalDictFromId(signal_id)
            signal_atoms = signal_dict['atoms']
            true_signal = np.zeros_like(signal_dict['signal'])
            for atom in signal_atoms:
                zs_atom = ZSAtom.from_dict(atom)
                zs_atom.padBothSides(self.dictionary.getAtomsLength())
                atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom['x'])
                true_signal += atom_signal    

            # Get the MMP approximation
            mmp_tree_dict = result['mmp-tree']
            mmp_approx_dict, mmp_approx_mse = self.getArgminMSEFromMMPTree(mmp_tree_dict)
            mmp_approx_atoms = mmp_approx_dict['atoms']

            # Get the OMP approximation
            omp_path_str = '-'.join(['1']*result['sparsity'])
            omp_approx_mse = mmp_tree_dict[omp_path_str]['mse']
            omp_approx_atoms = mmp_tree_dict[omp_path_str]['atoms']

            # Compute the reconstruction signals
            # Create the signals
            omp_signal = np.zeros_like(signal_dict['signal'])
            mmp_signal = np.zeros_like(signal_dict['signal'])

            for i, (omp_atom, mmp_atom) in enumerate(zip(omp_approx_atoms, mmp_approx_atoms)) :
                # Construct the atoms from parameters
                omp_zs_atom = ZSAtom(omp_atom['b'], omp_atom['y'], omp_atom['s'])
                omp_zs_atom.padBothSides(self.dictionary.getAtomsLength())
                mmp_zs_atom = ZSAtom(mmp_atom['b'], mmp_atom['y'], mmp_atom['s'])
                mmp_zs_atom.padBothSides(self.dictionary.getAtomsLength())
                # Get the atom signals
                omp_atom_signal = omp_zs_atom.getAtomInSignal(len(signal_dict['signal']), omp_atom['x'])
                omp_signal += omp_atom_signal
                mmp_atom_signal = mmp_zs_atom.getAtomInSignal(len(signal_dict['signal']), mmp_atom['x'])
                mmp_signal += mmp_atom_signal

            # Compute the reconstruction error
            omp_mse = np.mean((true_signal - omp_signal)**2)
            mmp_mse = np.mean((true_signal - mmp_signal)**2)

            # Get the prct of overlap type in the signal
            prct_overlap_type = self.signalOverlapTypePrctFromId(id=signal_id, nb_round=3)

            # Compute the local reconstruction error for each overlap interval
            for overlap_type, prct in prct_overlap_type.items():

                # Append the OMP data 
                data_prct_overlap_type['id'].append(signal_id)
                data_prct_overlap_type['snr'].append(signal_dict['snr'])
                data_prct_overlap_type['overlap_type'].append(overlap_type)
                data_prct_overlap_type['prct_in_signal'].append(prct)
                data_prct_overlap_type['algo_type'].append('OMP')
                data_prct_overlap_type['mse'].append(omp_mse)

                # Append the MMP data
                data_prct_overlap_type['id'].append(signal_id)
                data_prct_overlap_type['snr'].append(signal_dict['snr'])
                data_prct_overlap_type['overlap_type'].append(overlap_type)
                data_prct_overlap_type['prct_in_signal'].append(prct)
                data_prct_overlap_type['algo_type'].append('MMP-DF')
                data_prct_overlap_type['mse'].append(omp_mse)
            
        return data_prct_overlap_type

    def plotMMPDFLocalMSEOverlapPrctLineplot(self, mmpdf_db_path:str) :
        """
        Plot the lineplot of the MSE by overlap percentage 
        """
        plt.figure(figsize=(12, 8))

        overlap_prct_metrics = self.computeMMPDFLocalMSEPOverlapPrct(mmpdf_db_path)
        metrics_df = pd.DataFrame(overlap_prct_metrics)

        # Define colors and markers for the plots
        colors = {'OMP': 'navy', 'MMP-DF': 'red'}
        markers = {'>=1': 'o', '>=2': 'D', '>=3': 'X', '>=4': 's', '>=5': 'P'}

        # Group data and plot
        grouped = metrics_df.groupby(['algo_type', 'overlap_type'])
        for (algo_type, overlap_type), group in grouped:
            sns.lineplot(x='prct_in_signal', y='mse', data=group,
                        label=f'{algo_type} overlap {overlap_type}',
                        color=colors[algo_type],
                        marker=markers[overlap_type],
                        markersize=8)

        plt.title(f'OMP and MMP-DF MSE per number of minimum overlapped atom on the same interval', fontsize=16)
        plt.xlabel('Minimum number of overlapped atoms on the same interval', fontsize=14)
        plt.ylabel('MSE', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Algorithm & Sparsity', loc='upper left', fontsize=12)
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        plt.show()

    def plotMMPDFLocalMSEOverlapPrctLineplot2(self, mmpdf_db_path):
        """
        Plot the lineplot of the MSE by overlap percentage for OMP and MMP-DF algorithms.
        """
        # Suppose the following function returns the DataFrame as expected
        overlap_prct_metrics = self.computeMMPDFLocalMSEPOverlapPrct(mmpdf_db_path)
        metrics_df = pd.DataFrame(overlap_prct_metrics)

        plt.figure(figsize=(12, 8))
        markers = {'>=1': 'o', '>=2': 'D', '>=3': 'X', '>=4': 's', '>=5': 'P'}

        # Utilizing sns.scatterplot to plot data points with clear distinctions
        sns.scatterplot(data=metrics_df, x='prct_in_signal', y='mse', hue='algo_type',
                        style='overlap_type', markers=markers, palette=['navy', 'red'], s=100)

        plt.title('OMP and MMP-DF MSE per Overlap Type Percentage in the Signal', fontsize=16)
        plt.xlabel('Percentage of Overlap Type in Signal (%)', fontsize=14)
        plt.ylabel('MSE', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Algorithm & Overlap Type', loc='upper left', fontsize=12)
        plt.grid(True)
        plt.show()

    def plotMMPRankDistribution(self, db_path:str):
        """
        Plot the rank distribution of the MMP that has a better MSE score than OMP.
        The plot is sorted by sparsity level.
        """
        # Load the data
        with open(db_path, 'r') as f:
            data = json.load(f)

        # Filter the data to keep only the MMP results with better MSE than OMP
        mmp_results = [mmp_path for result in data['mmp'] for mmp_path, mmp_dict in result['mmp-tree'].items() if mmp_dict['mse'] < mmp_dict['-'.join([1 for _ in range(result['sparsity'])])]['mse']]
        
        
        # Sort the results by sparsity level
        sorted_results = sorted(mmp_results, key=lambda x: x['sparsity'])

        # Extract the ranks of the MMPs
        ranks = [result['rank'] for result in sorted_results]

        # Plot the rank distribution
        plt.figure(figsize=(12, 8))
        sns.histplot(ranks, bins=len(ranks), kde=True, color='blue')
        plt.title('Rank Distribution of MMPs with Better MSE than OMP', fontsize=16)
        plt.xlabel('Rank', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.show()

    def computeMMPRankDistribution(self, db_path:str) -> Dict:
        """
        Compute the MSE and the overlap percentage for each signal.
        Each row corresponds to a signal.
        """
        # Load the data
        with open(db_path, 'r') as f:
            output_data = json.load(f)

        data_prct_overlap_type = {
            'id': [],
            'snr': [],
            'overlap_type': [],
            'prct_in_signal': [],
            'algo_type': [],
            'mse': []
        }
        # Iterate over the outputs
        for result in output_data['mmp'] :

            # Reconstruct the denoised signal
            signal_id = result['id']
            signal_dict = self.signalDictFromId(signal_id)
            signal_atoms = signal_dict['atoms']
            true_signal = np.zeros_like(signal_dict['signal'])
            for atom in signal_atoms:
                zs_atom = ZSAtom.from_dict(atom)
                zs_atom.padBothSides(self.dictionary.getAtomsLength())
                atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom['x'])
                true_signal += atom_signal    

            # Get the MMP approximation
            mmp_tree_dict = result['mmp-tree']
            mmp_approx_dict, mmp_approx_mse = self.getArgminMSEFromMMPTree(mmp_tree_dict)
            mmp_approx_atoms = mmp_approx_dict['atoms']

            # Get the OMP approximation
            omp_path_str = '-'.join(['1']*result['sparsity'])
            omp_approx_mse = mmp_tree_dict[omp_path_str]['mse']
            omp_approx_atoms = mmp_tree_dict[omp_path_str]['atoms']

            # Compute the reconstruction signals
            # Create the signals
            omp_signal = np.zeros_like(signal_dict['signal'])
            mmp_signal = np.zeros_like(signal_dict['signal'])

            for i, (omp_atom, mmp_atom) in enumerate(zip(omp_approx_atoms, mmp_approx_atoms)) :
                # Construct the atoms from parameters
                omp_zs_atom = ZSAtom(omp_atom['b'], omp_atom['y'], omp_atom['s'])
                omp_zs_atom.padBothSides(self.dictionary.getAtomsLength())
                mmp_zs_atom = ZSAtom(mmp_atom['b'], mmp_atom['y'], mmp_atom['s'])
                mmp_zs_atom.padBothSides(self.dictionary.getAtomsLength())
                # Get the atom signals
                omp_atom_signal = omp_zs_atom.getAtomInSignal(len(signal_dict['signal']), omp_atom['x'])
                omp_signal += omp_atom_signal
                mmp_atom_signal = mmp_zs_atom.getAtomInSignal(len(signal_dict['signal']), mmp_atom['x'])
                mmp_signal += mmp_atom_signal

            # Compute the reconstruction error
            omp_mse = np.mean((true_signal - omp_signal)**2)
            mmp_mse = np.mean((true_signal - mmp_signal)**2)

            # Get the prct of overlap type in the signal
            prct_overlap_type = self.signalOverlapTypePrctFromId(id=signal_id, nb_round=3)

            # Compute the local reconstruction error for each overlap interval
            for overlap_type, prct in prct_overlap_type.items():

                # Append the OMP data 
                data_prct_overlap_type['id'].append(signal_id)
                data_prct_overlap_type['snr'].append(signal_dict['snr'])
                data_prct_overlap_type['overlap_type'].append(overlap_type)
                data_prct_overlap_type['prct_in_signal'].append(prct)
                data_prct_overlap_type['algo_type'].append('OMP')
                data_prct_overlap_type['mse'].append(omp_mse)

                # Append the MMP data
                data_prct_overlap_type['id'].append(signal_id)
                data_prct_overlap_type['snr'].append(signal_dict['snr'])
                data_prct_overlap_type['overlap_type'].append(overlap_type)
                data_prct_overlap_type['prct_in_signal'].append(prct)
                data_prct_overlap_type['algo_type'].append('MMP-DF')
                data_prct_overlap_type['mse'].append(omp_mse)
            
        return data_prct_overlap_type
    
                                                                                          
# 8 8888     ,o888888o.           .8.            d888888o.      d888888o.   8 888888888o   
# 8 8888    8888     `88.        .888.         .`8888:' `88.  .`8888:' `88. 8 8888    `88. 
# 8 8888 ,8 8888       `8.      :88888.        8.`8888.   Y8  8.`8888.   Y8 8 8888     `88 
# 8 8888 88 8888               . `88888.       `8.`8888.      `8.`8888.     8 8888     ,88 
# 8 8888 88 8888              .8. `88888.       `8.`8888.      `8.`8888.    8 8888.   ,88' 
# 8 8888 88 8888             .8`8. `88888.       `8.`8888.      `8.`8888.   8 888888888P'  
# 8 8888 88 8888            .8' `8. `88888.       `8.`8888.      `8.`8888.  8 8888         
# 8 8888 `8 8888       .8' .8'   `8. `88888.  8b   `8.`8888. 8b   `8.`8888. 8 8888         
# 8 8888    8888     ,88' .888888888. `88888. `8b.  ;8.`8888 `8b.  ;8.`8888 8 8888         
# 8 8888     `8888888P'  .8'       `8. `88888. `Y8888P ,88P'  `Y8888P ,88P' 8 8888         
    
    def plotAtomsPosisitonMatchingFromId(self, mmpdf_db_path:str, id:int) -> None :

        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)
            mmp_result_dict = next((result for result in output_data['mmp'] if result['id'] == id), None)

        # Get the true signal
        signal_dict = self.signalDictFromId(id)
        true_atoms = signal_dict['atoms']

        # Find the OMP and the MMP dict
        min_mse = np.inf
        mmp_dict = None
        mmp_path = None
        mmp_tree_dict = mmp_result_dict['mmp-tree']

        for path_str, leaf_dict in mmp_tree_dict.items() :
            if all(c == '1' for c in path_str.split('-')) :
                omp_dict = leaf_dict
            if leaf_dict['mse'] <= min_mse :
                mmp_dict = leaf_dict
                min_mse = leaf_dict['mse']
                mmp_path = path_str

        # Extract the atoms from the dict
        results_dict = [omp_dict, mmp_dict]
        results_name = ['OMP', f'MMP {mmp_path}']

        fig, axs = plt.subplots(len(true_atoms), 2, figsize=(15,3*len(true_atoms)), sharex=True)

        omp_matched_atoms = self.computeMatchingPosition(true_atoms, omp_dict['atoms'])
        mmp_matched_atoms = self.computeMatchingPosition(true_atoms, mmp_dict['atoms'])

        for i, (omp_matching, mmp_matching) in enumerate(zip(omp_matched_atoms, mmp_matched_atoms)) :
            true_atom, omp_atom = omp_matching
            true_atom, mmp_atom = mmp_matching

            corr_omp = self.dictionary.correlationFromDicts(true_atom, omp_atom, len(signal_dict['signal']))
            corr_mmp = self.dictionary.correlationFromDicts(true_atom, mmp_atom, len(signal_dict['signal']))

            # Build the true atom signal
            zs_true_atom = ZSAtom.from_dict(true_atom)
            zs_true_atom.padBothSides(self.dictionary.getAtomsLength())
            true_atom_signal = zs_true_atom.getAtomInSignal(len(signal_dict['signal']), true_atom['x'])

            # Build the OMP atom signal
            zs_omp_atom = ZSAtom.from_dict(omp_atom)
            zs_omp_atom.padBothSides(self.dictionary.getAtomsLength())
            omp_atom_signal = zs_omp_atom.getAtomInSignal(len(signal_dict['signal']), omp_atom['x'])

            # Build the MMP atom signal
            zs_mmp_atom = ZSAtom.from_dict(mmp_atom)
            zs_mmp_atom.padBothSides(self.dictionary.getAtomsLength())
            mmp_atom_signal = zs_mmp_atom.getAtomInSignal(len(signal_dict['signal']), mmp_atom['x'])

            # Compute the OMP correlation with the true signal
            (omp_correlation,) = np.correlate(true_atom_signal, omp_atom_signal, mode='valid')
            axs[i, 0].plot(true_atom_signal, label=f'True atom at x={true_atom["x"]}', color='g', lw=2, alpha=0.5)
            axs[i, 0].plot(omp_atom_signal, label=f'OMP atom at x={omp_atom["x"]}', color='b', lw=1)
            axs[i, 0].set_title(f'Atoms matching correlation = {omp_correlation}')
            axs[i, 0].legend(loc='best')
            axs[i, 0].axis('off')

            # Compute the MMP correlation with the true signal
            (mmp_correlation,) = np.correlate(true_atom_signal, mmp_atom_signal, mode='valid')
            axs[i, 1].plot(true_atom_signal, label=f'True atom at {true_atom["x"]}', color='g', lw=2, alpha=0.5)
            axs[i, 1].plot(mmp_atom_signal, label=f'MMP atom at {mmp_atom["x"]}', color='r', lw=1)
            axs[i, 1].set_title(f'Atoms matching correlation = {mmp_correlation}')
            axs[i, 1].legend(loc='best')
            axs[i, 1].axis('off')

        plt.axis('off')
        plt.show()

    def plotAtomsPosisitonMatchingFromDicts(self, signal_dict, true_atoms, approx_atoms, position_error_threshold:int=20, correlation_threshold:int=-1, verbose:bool=True) -> None :
        
        fig, axs = plt.subplots(len(true_atoms), 1, figsize=(15,3*len(true_atoms)), sharex=True)

        matched_atoms = self.computeMatchingPosition(true_atoms, approx_atoms)        

        for i, (true_atom, approx_atom) in enumerate(matched_atoms) :

            # Build the true atom signal
            zs_true_atom = ZSAtom.from_dict(true_atom)
            zs_true_atom.padBothSides(self.dictionary.getAtomsLength())
            true_atom_signal = zs_true_atom.getAtomInSignal(len(signal_dict['signal']), true_atom['x'])

            # Build the approx atom signal
            zs_approx_atom = ZSAtom.from_dict(approx_atom)
            zs_approx_atom.padBothSides(self.dictionary.getAtomsLength())
            approx_atom_signal = zs_approx_atom.getAtomInSignal(len(signal_dict['signal']), approx_atom['x'])

            # Compute the correlation with the true signal
            (corr,) = np.correlate(true_atom_signal, approx_atom_signal, mode='valid')
            axs[i, 0].plot(true_atom_signal, label=f'True atom at x={true_atom["x"]}', color='g', lw=2, alpha=0.5)
            axs[i, 0].plot(approx_atom_signal, label=f'Approx atom at x={approx_atom["x"]}', color='b', lw=1)
            axs[i, 0].set_title(f'Atoms matching correlation = {corr}')
            axs[i, 0].legend(loc='best')
            axs[i, 0].axis('off')

        plt.axis('off')
        plt.show()

    def computeCorrelationMatchingData(self, mmpdf_db_path:str) -> Dict:
        """
        Compute the position errors for all the signal on a MMPDF database
        """
        # Load the data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)

        data_errors = {
            'algorithm':[] ,
            'signal_id': [],
            'signal_snr': [],
            'signal_sparsness': [],
            'atoms_position_error': [],
            'atoms_correlation' : []
        }
        # Iterate over the outputs
        for result in output_data['mmp'] :

            # Get signal and approximation atoms
            signal_id = result['id']
            signal_dict = self.signalDictFromId(signal_id)
            true_atoms = signal_dict['atoms']

            # Get the MMPTree dict
            mmp_tree_dict = result['mmp-tree']
            # Find the OMP and the MMP dict
            min_mse = np.inf
            mmp_dict = None
            mmp_path = None
            for path_str, leaf_dict in mmp_tree_dict.items() :
                if all(c == '1' for c in path_str.split('-')) :
                    omp_dict = leaf_dict
                if leaf_dict['mse'] <= min_mse :
                    mmp_dict = leaf_dict
                    min_mse = leaf_dict['mse']
                    mmp_path = path_str

            omp_matched_atoms = self.computeMatchingPosition(true_atoms, omp_dict['atoms'])
            mmp_matched_atoms = self.computeMatchingPosition(true_atoms, mmp_dict['atoms'])

            for i, (omp_matching, mmp_matching) in enumerate(zip(omp_matched_atoms, mmp_matched_atoms)) :
                true_atom, omp_atom = omp_matching
                true_atom, mmp_atom = mmp_matching

                # OMP Metrics
                omp_corr = self.dictionary.correlationFromDicts(true_atom, omp_atom, len(signal_dict['signal']))
                omp_position_error = np.abs(true_atom['x'] - omp_atom['x'])

                # MMP Metrics
                mmp_corr = self.dictionary.correlationFromDicts(true_atom, mmp_atom, len(signal_dict['signal']))
                mmp_position_error = np.abs(true_atom['x'] - mmp_atom['x'])

                # Append the data for OMP
                data_errors['algorithm'].append('OMP')
                data_errors['signal_id'].append(signal_id)
                data_errors['signal_snr'].append(signal_dict['snr'])
                data_errors['signal_sparsness'].append(result['sparsity'])
                data_errors['atoms_position_error'].append(omp_position_error)
                data_errors['atoms_correlation'].append(omp_corr)

                # Append the data for MMP
                data_errors['algorithm'].append('MMP')
                data_errors['signal_id'].append(signal_id)
                data_errors['signal_snr'].append(signal_dict['snr'])
                data_errors['signal_sparsness'].append(result['sparsity'])
                data_errors['atoms_position_error'].append(mmp_position_error)
                data_errors['atoms_correlation'].append(mmp_corr)

        return data_errors

    def boxplotAtomCorrelationMatching(self, mmpdf_db_path:str, **kwargs) :
        """
        Plot the boxplot of the position errors.
        """
        plt.figure(figsize=(12, 8)) 
        legend_title = 'Algorithm'
        matching_data = self.computeCorrelationMatchingData(mmpdf_db_path)
        df = pd.DataFrame(matching_data)
        df = df.loc[df['signal_snr'] >= 0]
        for key, value in kwargs.items():
            if 'sparsity' in key.lower() or 'sparseness' in key.lower() :
                df = df.loc[df['signal_sparsness'] == value]
                legend_title += f' + sparsity={value} '
                
        sns.boxplot(x='signal_snr', y='atoms_correlation', hue='algorithm', data=df, palette="flare", fliersize=2, whis=1.5, showfliers=False)
        sns.despine(trim=True)
        plt.title(f'Correlation between matched atoms', fontsize=14)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=12)
        plt.ylabel('Sparsity', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title=legend_title, loc='best')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Set grid with partial transparency
        plt.show()

    def boxplotAtomPosErrorMatching(self, mmpdf_db_path:str, **kwargs) :
        """
        Plot the boxplot of the position errors.
        """
        plt.figure(figsize=(12, 8)) 
        legend_title = 'Algorithm'
        matching_data = self.computeCorrelationMatchingData(mmpdf_db_path)
        df = pd.DataFrame(matching_data)
        df = df.loc[df['signal_snr'] >= 0]
        for key, value in kwargs.items():
            if 'sparsity' in key.lower() or 'sparseness' in key.lower() :
                df = df.loc[df['signal_sparsness'] == value]
                legend_title += f' + sparsity={value} '
                
        sns.boxplot(x='signal_snr', y='atoms_position_error', hue='algorithm', data=df, palette="flare", fliersize=2, whis=1.5, showfliers=False)
        sns.despine(trim=True)
        plt.title(f'Position error between matched atoms', fontsize=14)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=12)
        plt.ylabel('Sparsity', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title=legend_title, loc='best')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Set grid with partial transparency
        plt.show()
        
    def plotPosErrAtStep(self, db_path:str, step:int) :
        """
        Plot the boxplot of the position errors.
        """
        plt.figure(figsize=(12, 8)) 
        data_errors = self.computeOMPPositionErrorsPerStep(db_path)
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

    def plotSignalOverlapFromId(self, id:int) -> None :
        """
        Plot the signal with a conditional coloring according to the overlap vector.
        """
        # Get the signal dictionary
        signal_dict = self.signalDictFromId(id)
        atoms_list = signal_dict['atoms']
        overlap_vector = self.getSignalOverlapVectorFromId(id)

        # Plot the signal
        fig, axs = plt.subplots(2, 1, figsize=(12, 2*2), sharex=True)
        axs[0].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
        true_signal = np.zeros_like(signal_dict['signal'])

        # Color the background based on overlap
        cmap = plt.get_cmap('plasma')
        max_overlap = max(overlap_vector)
        norm = Normalize(vmin=0, vmax=max_overlap)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for idx, val in enumerate(overlap_vector):
            axs[0].axvspan(idx, idx+1, facecolor=cmap(norm(val)), alpha=0.3)
            axs[1].axvspan(idx, idx+1, facecolor=cmap(norm(val)), alpha=0.3)

        # Compute the atoms signals
        offset = 1.5*max(np.abs(signal_dict['signal']))
        for i, atom in enumerate(atoms_list):
            zs_atom = ZSAtom.from_dict(atom)
            zs_atom.padBothSides(self.dictionary.getAtomsLength())
            atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom['x'])
            true_signal += atom_signal
            # Plot the atom's signal
            axs[1].plot(atom_signal + i*offset, label=f'Atom at {atom["x"]}', alpha=0.6, lw=2)

        axs[0].plot(true_signal, label='True signal', color='g', lw=2)   
        axs[0].legend(loc='best')
        axs[0].axis('off')
        #axs[1].legend(loc='best')
        axs[1].axis('off')

        # Add a colorbar
        fig.colorbar(sm, ax=axs, orientation='vertical', label='Overlap level', pad=0.01)
        plt.show()
    
    def plotMethodComparison(self, mmpdf_db_path:str, id:int) -> None :
        """
        Use three subplots to compare the results between the OMP and the MMP results.
        The OMP result corresponds to the first branch of the MMP tree.
        The MMP result corresponds to the MSE-argmin of the MMP tree.
        """
        # Load the data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)
            mmp_result_dict = next((result for result in output_data['mmp'] if result['id'] == id), None)
        
        # Get the true signal
        signal_dict = self.signalDictFromId(id)
        true_atoms = signal_dict['atoms']
        true_signal = np.zeros_like(signal_dict['signal'])
        for atom_dict in true_atoms :
            zs_atom = ZSAtom(atom_dict['b'], atom_dict['y'], atom_dict['s'])
            zs_atom.padBothSides(self.dictionary.getAtomsLength())
            atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom_dict['x'])
            true_signal += atom_signal

        fig, axs = plt.subplots(3, 1, figsize=(15, 4*3), sharex=True)
        axs[0].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
        axs[0].plot(true_signal, label='True signal', color='g')

        # Get the overlap vector of the signal
        overlap_vector = self.getSignalOverlapVectorFromId(id)

        # Create the color map
        cmap = plt.get_cmap('plasma')
        max_overlap = max(overlap_vector)
        norm = Normalize(vmin=0, vmax=max_overlap)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for idx, val in enumerate(overlap_vector):
            axs[0].axvspan(idx, idx+1, facecolor=cmap(norm(val)), alpha=0.2)
            axs[1].axvspan(idx, idx+1, facecolor=cmap(norm(val)), alpha=0.2)
            axs[2].axvspan(idx, idx+1, facecolor=cmap(norm(val)), alpha=0.2)
            
        # Find the OMP and the MMP dict
        min_mse = np.inf
        mmp_dict = None
        mmp_path = None
        mmp_tree_dict = mmp_result_dict['mmp-tree']

        for path_str, leaf_dict in mmp_tree_dict.items() :
            if all(c == '1' for c in path_str.split('-')) :
                omp_dict = leaf_dict
            if leaf_dict['mse'] <= min_mse :
                mmp_dict = leaf_dict
                min_mse = leaf_dict['mse']
                mmp_path = path_str

        # Extract the atoms from the dict
        results_dict = [omp_dict, mmp_dict]
        results_name = ['OMP', f'MMP {mmp_path}']

        # Plot the comparison
        for i, result_dict in enumerate(results_dict) :
            approx, _ = self.dictionary.getSignalProjectionFromAtoms(signal_dict['signal'], result_dict['atoms'])
            axs[0].plot(approx, color=f'C{i}', label=results_name[i])
            axs[i+1].plot(true_signal, color='g')
            axs[i+1].plot(approx, color=f'C{i}')
            axs[i+1].plot(signal_dict['signal'], color='k', alpha=0.4, lw=3)
            axs[i+1].set_title('{} : MSE = {}'.format(results_name[i], result_dict["mse"]), fontsize=12)
            axs[0].legend(loc='best')
            axs[i+1].axis('off')  

        axs[0].set_title('       True signal', fontsize=14, loc='left')
        axs[0].legend(loc='best') 
        axs[0].axis('off')

        # Add a colorbar
        cbar = fig.colorbar(sm, ax=axs, orientation='vertical', label='Overlap level', pad=0.01)
        cbar.solids.set(alpha=0.4)

        plt.show()
                                         
#       8 888888888o   8 888888888o.      ,o888888o.    
#       8 8888    `88. 8 8888    `88.    8888     `88.  
#       8 8888     `88 8 8888     `88 ,8 8888       `8. 
#       8 8888     ,88 8 8888     ,88 88 8888           
#       8 8888.   ,88' 8 8888.   ,88' 88 8888           
#       8 888888888P'  8 888888888P'  88 8888           
#       8 8888         8 8888`8b      88 8888           
#       8 8888         8 8888 `8b.    `8 8888       .8' 
#       8 8888         8 8888   `8b.     8888     ,88'  
#       8 8888         8 8888     `88.    `8888888P'    

    @staticmethod
    def preprocessPRCurveDatas(pr: np.ndarray) -> np.ndarray:
        """Preprocesses a precision-recall (PR) array for plotting by sorting the rows by
        ascending recall values and inserting endpoints at recall=0 and recall=1.

        This function is designed to prepare precision-recall data for smooth plotting and
        interpolation by ensuring that the recall values are sorted in ascending order and
        that the precision-recall curve includes endpoints at recall values of 0 and 1.

        Parameters
        ----------
        pr : np.ndarray
            A 2D numpy array with shape (n_pr, 2) where the first column contains precision
            values and the second column contains recall values.

        Returns
        -------
        np.ndarray
            A 2D numpy array with shape (n_pr + 2, 2) where the rows are sorted by ascending
            recall values, and additional rows are inserted at the beginning and end to
            include the endpoints at recall=0 and recall=1.

        Example
        -------
        >>> pr = np.array([[0.8, 0.1], [0.6, 0.4], [0.9, 0.3]])
        >>> preprocess_pr_for_plot(pr)
        array([[0.8, 0. ],
        [0.8, 0.1],
        [0.9, 0.3],
        [0.6, 0.4],
        [0.6, 1. ]])
        """
        prec, rec = pr.T

        # sort by ascending recall values
        sort_by_rec = np.argsort(rec)
        pr_sorted = pr[sort_by_rec]

        # insert point at recall=0
        first_prec, first_rec = pr_sorted[0, 0], 0
        pr_sorted = np.insert(
            arr=pr_sorted, obj=0, values=np.array([first_prec, first_rec]), axis=0
        )

        # insert point at recall=1
        last_prec, last_rec = pr_sorted[-1, 0], 1
        pr_sorted = np.insert(
            arr=pr_sorted,
            obj=pr_sorted.shape[0],
            values=np.array([last_prec, last_rec]),
            axis=0,
        )

        return pr_sorted

    @staticmethod
    def plotPRCurve(pr: np.ndarray, ax=None, **plot_kwargs):
        """
        Plots a precision-recall curve using the same conventions as scikit-learn.

        This function takes a precision-recall array and plots the corresponding precision-recall
        curve. The precision-recall data is first preprocessed to ensure that recall values
        are sorted and endpoints are added for smooth plotting.

        Parameters
        ----------
        pr : np.ndarray
            A 2D numpy array with shape (n_pr, 2) where the first column contains precision
            values and the second column contains recall values.
        ax : matplotlib.axes.Axes, optional
            The axes object to draw the plot onto, or None to create a new figure and axes.
        **plot_kwargs
            Additional keyword arguments to pass to the `ax.plot` function.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the precision-recall curve plotted.

        Example
        -------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> pr = np.array([[0.8, 0.1], [0.6, 0.4], [0.9, 0.3]])
        >>> ax = plot_pr_curve(pr)
        >>> plt.show()
        """
        if ax is None:
            fig, ax = plt.subplots()

        pr = CSCWorkbench.preprocessPRCurveDatas(pr)
        prec, rec = pr.T
        ax.plot(rec, prec, ls="--", drawstyle="steps-post", **plot_kwargs)
        ax.set(
            xlabel="Recall",
            xlim=(-0.01, 1.01),
            ylabel="Precision",
            ylim=(-0.01, 1.01),
            aspect="equal",
        )
        return ax
    
    @staticmethod
    def computeMeanPRCurve(all_pr, n_samples):
        """
        Computes the mean Precision-Recall (PR) curve along with the curves
        representing one standard deviation above and below the mean.

        Parameters:
        -----------
        all_pr : list of numpy.ndarray
            A list where each element is a numpy array of shape (n_pr, 2)
            representing precision and recall values.
        n_samples : int
            The desired number of samples for the approximation.

        Returns:
        --------
        pr_mean : numpy.ndarray
            A 2D array where the first column contains the mean precision values
            and the second column contains the corresponding recall values.
        pr_mean_plus_std : numpy.ndarray
            A 2D array where the first column contains the mean precision values
            plus one standard deviation and the second column contains the
            corresponding recall values.
        pr_mean_minus_std : numpy.ndarray
            A 2D array where the first column contains the mean precision values
            minus one standard deviation and the second column contains the
            corresponding recall values.

        Notes:
        ------
        This function interpolates the precision values for a common recall axis
        ranging from 0 to 1, then computes the mean and standard deviation of
        these interpolated precision values across all provided PR curves.
        """
        n_pr = len(all_pr)
        recall_axis = np.linspace(0, 1, n_samples, endpoint=True)

        all_approx = np.empty((n_pr, n_samples))

        for k_pr in range(n_pr):
            pr = all_pr[k_pr]
            p, r = CSCWorkbench.preprocessPRCurveDatas(pr).T
            approx_fun = interp1d(r, p, kind="previous")
            approx = approx_fun(recall_axis)
            all_approx[k_pr] = approx
            #CSCWorkbench.plotPRCurve(pr, ax=ax, color="b", alpha=0.2)

        prec_mean = all_approx.mean(0)
        prec_std = all_approx.std(0)
        prec_mean_plus_std = prec_mean + prec_std
        prec_mean_minus_std = prec_mean - prec_std

        pr_mean = np.c_[prec_mean, recall_axis]
        pr_mean_plus_std = np.c_[prec_mean_plus_std, recall_axis]
        pr_mean_minus_std = np.c_[prec_mean_minus_std, recall_axis]

        return pr_mean, pr_mean_plus_std, pr_mean_minus_std


    
#    /$$$$$$$  /$$$$$$$   /$$$$$$        /$$$$$$$  /$$                     /$$ /$$                    
#   | $$__  $$| $$__  $$ /$$__  $$      | $$__  $$|__/                    | $$|__/                    
#   | $$  \ $$| $$  \ $$| $$  \__/      | $$  \ $$ /$$  /$$$$$$   /$$$$$$ | $$ /$$ /$$$$$$$   /$$$$$$ 
#   | $$$$$$$/| $$$$$$$/| $$            | $$$$$$$/| $$ /$$__  $$ /$$__  $$| $$| $$| $$__  $$ /$$__  $$
#   | $$____/ | $$__  $$| $$            | $$____/ | $$| $$  \ $$| $$$$$$$$| $$| $$| $$  \ $$| $$$$$$$$
#   | $$      | $$  \ $$| $$    $$      | $$      | $$| $$  | $$| $$_____/| $$| $$| $$  | $$| $$_____/
#   | $$      | $$  | $$|  $$$$$$/      | $$      | $$| $$$$$$$/|  $$$$$$$| $$| $$| $$  | $$|  $$$$$$$
#   |__/      |__/  |__/ \______/       |__/      |__/| $$____/  \_______/|__/|__/|__/  |__/ \_______/
#                                                     | $$                                            
#                                                     | $$                                            
#                                                     |__/                                            


    #
    #             888           888                      e88~-_  ,d88~~\  e88~-_  
    #     /~~~8e  888 888-~88e  888-~88e   /~~~8e       d888   \ 8888    d888   \ 
    #         88b 888 888  888b 888  888       88b ____ 8888     `Y88b   8888     
    #    e88~-888 888 888  8888 888  888  e88~-888      8888      `Y88b, 8888     
    #   C888  888 888 888  888P 888  888 C888  888      Y888   /    8888 Y888   / 
    #    "88_-888 888 888-_88"  888  888  "88_-888       "88_-~  \__88P'  "88_-~  
    #                 888                                                         
    #  

    def plotAlphaCSCDecomposition(self, signal_dict:dict, sparsity:int, verbose:bool=False) :
        """
        Plot the alphaCSC decomposition of a signal
        Args :
            signal_dict (dict) : The signal dictionary
            sparsity (int) : The sparsity level
            verbose (bool) : The verbosity flag
        """
        # Retrieve the signal from the datas
        signal = signal_dict['signal']
        atoms = self.dictionary.alphaCSCResultFromDict(signal_dict, nb_activations=sparsity)
        atoms = atoms[:sparsity]

        if verbose :
            print(f"plot alphaCSC decomposition for {signal_dict['id']} with {sparsity} atoms")

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(signal, label='Noisy signal', color='k', lw=3, alpha=0.4)

        approx = np.zeros_like(signal)
        offset = max(np.abs(np.array(signal))) * 0.7
        for i, atom in enumerate(atoms) :
            zs_atom = ZSAtom.from_dict(atom)
            zs_atom.padBothSides(self.dictionary.getAtomsLength())
            atom_signal = zs_atom.getAtomInSignal(signal_length=len(signal), offset=atom['x'])
            approx += atom_signal
            ax.plot(atom_signal - (i+1)*offset, label=f'Atom n°{i+1}', alpha=0.7)
        
        ax.plot(approx, label='Approx')
        plt.title(f'alphaCSC Decomposition with {sparsity} atoms', fontsize=16)
        plt.legend(loc='best', title='Components')
        plt.show()

    def alphaCSCPRCurve(self, signal_dict:dict, n_samples:int, pos_err_threshold:int, corr_err_threshold:float, verbose:bool=False) :
        """Compute the PR curve for the Alpha-CSC algorithm
        Args:
            signal_dict (dict) : The signal dictionary
            n_samples (int) : The number of samples for the PR curve
            pos_err_threshold (int) : The position error threshold
            corr_err_threshold (float) : The correlation error threshold
            verbose (bool) : The verbosity flag
        Returns:
            pr_curve (np.ndarray) : The PR curve
        """
        signal = np.array(signal_dict['signal'])
        D = self.dictionary.getLocalDictionary()

        if signal.ndim == 1:
            signal = signal[np.newaxis, :]

        lmbda = 8e-4 # initial small lambda
        activations = []
        target_activations = 5
        max_nb_activations = 1000
        min_nb_activations = 1
        prct_tolerance = 30

        pr_metrics = []
        iter = 0

        if verbose : 
            print(f'Processing signal {signal_dict["id"]}')
            print(f'First lambda = {lmbda:.2e} : len pr_metrics = {len(pr_metrics)} & n_samples = {n_samples}')

        while len(pr_metrics) < n_samples :

            activations = update_z(signal, D, lmbda).squeeze()  # Assuming update_z returns the activations array
            activations = activations.flatten()
            nnz_indexes, = np.nonzero(activations)
            len_activations = len(nnz_indexes)

            if verbose:
                print(f'Iteration {iter+1}: lambda = {lmbda:.2e}, number of activations = {len_activations}')

            if len_activations <= max_nb_activations and len_activations >= min_nb_activations :

                # Post-processing of activations
                nnz_values = activations[nnz_indexes]
                order = np.argsort(nnz_values)[::-1]
                nnz_indexes_sorted = nnz_indexes[order]

                # Extract atoms and their parameters
                positions_idx, atoms_idx = np.unravel_index(nnz_indexes_sorted, shape=activations.reshape(-1, len(D)).shape)

                approx_atoms = list()
                for pos_idx, atom_idx in zip(positions_idx, atoms_idx):
                    b, y, s = self.dictionary.atoms[atom_idx].params['b'], self.dictionary.atoms[atom_idx].params['y'], self.dictionary.atoms[atom_idx].params['sigma']
                    approx_atoms.append({'x': pos_idx, 'b': b, 'y': y, 's': s})

                # Compute the PR metrics
                tp = self.computeMaxTruePositives(signal_dict['atoms'], approx_atoms, pos_err_threshold, corr_err_threshold)
                precision = tp / len(approx_atoms)
                recall = tp / len(signal_dict['atoms'])
                pr_metrics.append([precision, recall])

                if verbose :
                    print(f' => Added PR metrics : TP = {tp}, precision = {tp}/{len(approx_atoms)}={precision}, recall = {tp}/{len(signal_dict["atoms"])}={recall}')

            # Update lambda based on the difference between current and target activations
            if len_activations > target_activations:
                last_lmbda = lmbda
                lmbda *= 1 + 0.01*(len_activations - target_activations) / target_activations
            else:
                last_lmbda_coeff = 0.98
                lmbda = (1 - last_lmbda_coeff) * lmbda + last_lmbda_coeff * last_lmbda

            iter += 1
            
        pr_curve = np.array(pr_metrics)
        return pr_curve

    def alphaCSCResultFromDict(self, signal_dict:dict, n_samples:int, pos_err_threshold:int, corr_err_threshold:float, verbose:bool=False) :
        """Compute the PR curve for the Alpha-CSC algorithm
        Args:
            signal_dict (dict) : The signal dictionary
            n_samples (int) : The number of samples for the PR curve
            sparsity
            pos_err_threshold (int) : The position error threshold
            corr_err_threshold (float) : The correlation error threshold
            verbose (bool) : The verbosity flag
        Returns:
            atoms (List[dict]) : The atoms that maximize the TP value
        """
        signal = np.array(signal_dict['signal'])
        D = self.dictionary.getLocalDictionary()

        if signal.ndim == 1:
            signal = signal[np.newaxis, :]

        lmbda = 8e-4 # initial small lambda
        activations = []
        target_activations = len(signal_dict['atoms'])
        max_nb_activations = 100
        min_nb_activations = 1

        pr_metrics = []
        iter = 0

        list_tp = []
        list_atoms = []

        if verbose : 
            print(f'Processing signal {signal_dict["id"]}')

        while len(list_tp) < n_samples :

            activations = update_z(signal, D, lmbda).squeeze()  # Assuming update_z returns the activations array
            activations = activations.flatten()
            nnz_indexes, = np.nonzero(activations)
            len_activations = len(nnz_indexes)

            if verbose:
                print(f'Iteration {iter+1}: lambda = {lmbda:.2e}, number of activations = {len_activations}')

            if len_activations <= max_nb_activations and len_activations >= min_nb_activations :

                # Post-processing of activations
                nnz_values = activations[nnz_indexes]
                order = np.argsort(nnz_values)[::-1]
                nnz_indexes_sorted = nnz_indexes[order]

                # Extract atoms and their parameters
                positions_idx, atoms_idx = np.unravel_index(nnz_indexes_sorted, shape=activations.reshape(-1, len(D)).shape)

                approx_atoms = list()
                for pos_idx, atom_idx in zip(positions_idx, atoms_idx):
                    b, y, s = self.dictionary.atoms[atom_idx].params['b'], self.dictionary.atoms[atom_idx].params['y'], self.dictionary.atoms[atom_idx].params['sigma']
                    approx_atoms.append({'x': pos_idx, 'b': b, 'y': y, 's': s})

                # Compute the PR metrics
                tp = self.computeMaxTruePositives(signal_dict['atoms'], approx_atoms, pos_err_threshold, corr_err_threshold)
                list_tp.append(tp)
                list_atoms.append(approx_atoms)

                if verbose :
                    print(f'    {len(list_tp)}/{n_samples} => {len(approx_atoms)} new atoms append to results')
            
            # Update lambda based on the difference between current and target activations
            if len_activations > target_activations:
                last_lmbda = lmbda
                lmbda *= 1 + 0.01*(len_activations - target_activations) / target_activations
            else:
                last_lmbda_coeff = 0.98
                lmbda = (1 - last_lmbda_coeff) * lmbda + last_lmbda_coeff * last_lmbda

            iter += 1

        max_tp = max(list_tp)
        max_tp_indexes = [i for i, tp in enumerate(list_tp) if tp == max_tp]

        if verbose:
            print(f'    List tp for samples: {list_tp}')
            print(f'    Max tp value: {max_tp}')
            print(f'    Indices with max tp: {max_tp_indexes}')

        # Find the shortest list of atoms among those with the max tp
        approx_atoms = min([list_atoms[i] for i in max_tp_indexes], key=len)
        
        if verbose:
            print(f'    Shortest list among max tp: {approx_atoms}')

        results_dict = {
            'id': signal_dict['id'],
            'snr': signal_dict['snr'],
            'atoms': approx_atoms
        }
        
        return results_dict

    def alphaCSCPipelineFromSignalsDB(self, output_filename:str, nb_cores:int, n_samples:int, pos_err_threshold:int, corr_err_threshold:float, verbose:bool=False) :
        """Create a pipeline of the AlphaCSC-L1 algorithm from the database of signals.
        Args:
            input_filename (str): The name of the input file containing the signals database
            output_filename (str): The name of the output file to store the results
        Returns:
            None : it saves the results in a file
        """
        with open(self.signals_path, 'r') as json_file:
            data = json.load(json_file)
            if data is None:
                raise ValueError("The input file is empty or does not contain any data.")
        
        if verbose :
            print(f"AlphaCSC Pipeline from {self.signals_path} with {len(data['signals'])} signals")

        # Extract the signals from the DB
        signals = data['signals']
        # Create the results dictionary
        results = dict()
        results['source'] = self.signals_path
        results['date'] = get_today_date_str()
        results['algorithm'] = 'Convolutional MMP-DF'
        results['nbSamples'] = n_samples
        results['posErrThreshold'] = pos_err_threshold
        results['corrErrThreshold'] = corr_err_threshold
        results['batchSize'] = data['batchSize']
        results['snrLevels'] = data['snrLevels']
        results['signalLength'] = data['signalLength']
        results['sparsityLevels'] = data['sparsityLevels']
        results['dictionary'] = str(self)

        # Parallelize the OMP algorithm on the signals from the DB
        mmpdf_results = Parallel(n_jobs=nb_cores)(delayed(self.alphaCSCResultFromDict)(signal_dict, n_samples=n_samples, pos_err_threshold=pos_err_threshold, corr_err_threshold=corr_err_threshold, verbose=verbose) for signal_dict in tqdm(signals, desc='AlphaCSC Pipeline from DB'))
        results['results'] = mmpdf_results
        # Save the results in a JSON file
        json.dump(results, open(output_filename, 'w'), indent=4, default=handle_non_serializable)
        if verbose :
            print(f"AlphaCSC Pipeline results saved in {output_filename}")
                                                                
    #         ________  ________  ________  ________  ___      ___ ________  ________     
    #       |\   ____\|\   __  \|\   __  \|\   __  \|\  \    /  /|\   __  \|\   __  \    
    #        \ \  \___|\ \  \|\  \ \  \|\  \ \  \|\  \ \  \  /  / | \  \|\  \ \  \|\  \   
    #         \ \_____  \ \   ____\ \   __  \ \   _  _\ \  \/  / / \ \   __  \ \   _  _\  
    #          \|____|\  \ \  \___|\ \  \ \  \ \  \\  \\ \    / /   \ \  \ \  \ \  \\  \| 
    #            ____\_\  \ \__\    \ \__\ \__\ \__\\ _\\ \__/ /     \ \__\ \__\ \__\\ _\ 
    #           |\_________\|__|     \|__|\|__|\|__|\|__|\|__|/       \|__|\|__|\|__|\|__|
    #           \|_________|                                                              
                                                                                                                   
    def computeSparVarMetricsFromDict(self, sparvar_dict:dict, pos_err_threshold:int=20, corr_err_threshold:float=0.75) -> Tuple[List[float], List[float]] :
        """
        Compute the precisions-recalls metrics from a sparsity variation dictionary
        """

        # Initialize the precion-recall lists
        pr_metrics = []

        # Extract the signal id, snr and results
        signal_id = sparvar_dict['id']
        signal_snr = sparvar_dict['snr']
        results_list = sparvar_dict['results']

        # Get the true atoms from the signal
        signal_dict = self.signalDictFromId(signal_id)
        true_atoms = signal_dict['atoms']
        nb_true_atoms = len(true_atoms)

        # Iterate over the results
        for result_dict in results_list :
            
            approx_atoms = result_dict['atoms']
            np_approx_atoms = len(approx_atoms)

            # Compute the true positives, precision and recall
            tp = self.computeMaxTruePositives(true_atoms, approx_atoms, pos_err_threshold, corr_err_threshold)
            precision = tp / np_approx_atoms
            recall = tp / nb_true_atoms

            pr_metrics.append([precision, recall])

        return np.array(pr_metrics)
    
    def computeSparVarMetricsFromDB(self, sparvar_db_path:str, results_key:str, pos_err_threshold:int, corr_err_threshold:float, sparsity_criteria:int, snr_criteria:int, verbose:bool=False) -> Tuple[List[float], List[float]] :
        """
        Compute the precisions-recalls metrics from a sparsity variation database
        Args :
            sparvar_db_path (str) : The path to the sparsity variation database
            results_key (str) : The key of the results in the database
            pos_err_threshold (int) : The position error threshold
            corr_err_threshold (float) : The correlation error threshold
            verbose (bool) : The verbosity flag
        Returns :
            pr_mean : numpy.ndarray
                A 2D array where the first column contains the mean precision values
                and the second column contains the corresponding recall values.
            pr_mean_plus_std : numpy.ndarray
                A 2D array where the first column contains the mean precision values
                plus one standard deviation and the second column contains the
                corresponding recall values.
            pr_mean_minus_std : numpy.ndarray
                A 2D array where the first column contains the mean precision values
                minus one standard deviation and the second column contains the
                corresponding recall values.
        """
        with open(sparvar_db_path, 'r') as f:
            sparvar_db = json.load(f)
            sparvar_results = sparvar_db[results_key]
            max_sparsity = sparvar_db['maxSparsityLevel']

        def signal_condition(sparsity, snr) :
            if sparsity_criteria == -1 and snr_criteria != -1 :
                return (snr == snr_criteria)
            elif sparsity_criteria != -1 and snr_criteria == -1 :
                return (sparsity == sparsity_criteria)
            elif sparsity_criteria != -1 and snr_criteria != -1 :
                return (sparsity == sparsity_criteria) and (snr == snr_criteria)
            return True

        pr_results = [] # List of np.array of [precision, recall] metrics
        for result_dict in sparvar_results :
            signal_id = result_dict['id']
            signal_dict = self.signalDictFromId(signal_id)
            signal_sparsity, signal_snr = len(signal_dict['atoms']), signal_dict['snr']
            if signal_condition(signal_sparsity, signal_snr) :
                pr_array = self.computeSparVarMetricsFromDict(
                    result_dict,
                    pos_err_threshold,
                    corr_err_threshold
                )
                pr_results.append(pr_array)

        pr_mean, pr_mean_plus_std, pr_mean_minus_std = CSCWorkbench.computeMeanPRCurve(pr_results, max_sparsity)
        return pr_mean, pr_mean_plus_std, pr_mean_minus_std

    def plotPRCurvesFromDB(self, **kwargs) :
        """
        Plot the precision-recall curves from the results of the any sparVar database.
        """
        pos_err_threshold = 20
        corr_err_threshold = 0.75
        sparvar_db_paths = {}
        verbose = False
        fill = False
        snr_criteria = -1
        sparsity_criteria = -1

        for key, value in kwargs.items() :
            if key == 'pos_err_threshold' :
                pos_err_threshold = value
            elif key == 'corr_err_threshold' :
                corr_err_threshold = value
            elif key == 'sparsity_criteria' :
                sparsity_criteria = value
            elif key == 'snr_criteria' :
                snr_criteria = value
            elif key == 'verbose' :
                verbose = value
            elif key == 'fill' :
                fill = value
            else :
                sparvar_db_paths[str(key).lower()] = value

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        for i, (algorithm, path) in enumerate(sparvar_db_paths.items()) :
            pr_mean, pr_mean_plus_std, pr_mean_minus_std = self.computeSparVarMetricsFromDB(
                sparvar_db_path = path,
                results_key = str(algorithm),
                pos_err_threshold = pos_err_threshold,
                corr_err_threshold = corr_err_threshold,
                sparsity_criteria = sparsity_criteria,
                snr_criteria = snr_criteria,
                verbose=verbose
                )
            CSCWorkbench.plotPRCurve(pr_mean, ax=ax, color=f'C{i}', label='CSC-'+str(algorithm).upper())
            CSCWorkbench.plotPRCurve(pr_mean_plus_std, ax=ax, color=f'C{i}', alpha=0.3)
            CSCWorkbench.plotPRCurve(pr_mean_minus_std, ax=ax, color=f'C{i}', alpha=0.3)
            if fill :
                plt.fill_between(pr_mean[:, 1], pr_mean_plus_std[:, 0], pr_mean_minus_std[:, 0], alpha=0.1)

        plt.title(f'Precision-Recall Curve : pos_err_threshold = {pos_err_threshold} & corr_err_threshold = {corr_err_threshold}', fontsize=16)
        plt.legend(loc='best', title='Algorithm')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.grid(alpha=0.5)
        plt.show() 

                                                        
    #       ,o888888o.    8 888888888o.      8 888888888o.      
    #      8888     `88.  8 8888    `^888.   8 8888    `^888.   
    #   ,8 8888       `8. 8 8888        `88. 8 8888        `88. 
    #   88 8888           8 8888         `88 8 8888         `88 
    #   88 8888           8 8888          88 8 8888          88 
    #   88 8888           8 8888          88 8 8888          88 
    #   88 8888           8 8888         ,88 8 8888         ,88 
    #   `8 8888       .8' 8 8888        ,88' 8 8888        ,88' 
    #      8888     ,88'  8 8888    ,o88P'   8 8888    ,o88P'   
    #       `8888888P'    8 888888888P'      8 888888888P'      

    def computeCDCTruePositivesFromDB(self, sparvar_db_path:str, results_key:str, pos_err_threshold:int, corr_err_threshold:float, sparsity_criteria:int, snr_criteria:int, verbose:bool=False) -> Tuple[List[float], List[float]] :
        """
        Compute the precisions-recalls metrics from a sparsity variation database
        Args :
            sparvar_db_path (str) : The path to the sparsity variation database
            results_key (str) : The key of the results in the database
            pos_err_threshold (int) : The position error threshold
            corr_err_threshold (float) : The correlation error threshold
            verbose (bool) : The verbosity flag
        Returns :
            tp_results : List[float] : The list of true positives metrics
        """
        with open(sparvar_db_path, 'r') as f:
            sparvar_db = json.load(f)
            sparvar_results = sparvar_db[results_key]

        def signal_condition(sparsity, snr) :
            if sparsity_criteria == -1 and snr_criteria != -1 :
                return (snr == snr_criteria)
            elif sparsity_criteria != -1 and snr_criteria == -1 :
                return (sparsity == sparsity_criteria)
            elif sparsity_criteria != -1 and snr_criteria != -1 :
                return (sparsity == sparsity_criteria) and (snr == snr_criteria)
            return True

        tp_results = [] # List of int of true_positives metrics
        # Iterate over the result_dict of each signal in the sparvar_results
        for result_dict in sparvar_results :
            signal_id = result_dict['id']
            signal_dict = self.signalDictFromId(signal_id)
            signal_sparsity, signal_snr = len(signal_dict['atoms']), signal_dict['snr']
            # Check if the signal satisfies the criteria
            if signal_condition(signal_sparsity, signal_snr) :
                true_atoms = signal_dict['atoms']
                approx_atoms = result_dict['results'][signal_sparsity-1]['atoms']
                tp = self.computeMaxTruePositives(true_atoms, approx_atoms, pos_err_threshold, corr_err_threshold)
                tp_results.append(tp)

        return tp_results
    
    def computeCDCTruePositivesFromAlphaCSC(self, alphaCSC_db_path:str, results_key:str, pos_err_threshold:int, corr_err_threshold:float, sparsity_criteria:int, snr_criteria:int, verbose:bool=False) -> Tuple[List[float], List[float]] :
        """
        Compute the true positives metrics from a alpha CSC results database
        Args :
            sparvar_db_path (str) : The path to the sparsity variation database
            results_key (str) : The key of the results in the database
            pos_err_threshold (int) : The position error threshold
            corr_err_threshold (float) : The correlation error threshold
            verbose (bool) : The verbosity flag
        Returns :
            tp_results : List[float] : The list of true positives metrics
        """
        with open(alphaCSC_db_path, 'r') as f:
            sparvar_db = json.load(f)
            sparvar_results = sparvar_db[results_key]

        def signal_condition(sparsity, snr) :
            if sparsity_criteria == -1 and snr_criteria != -1 :
                return (snr == snr_criteria)
            elif sparsity_criteria != -1 and snr_criteria == -1 :
                return (sparsity == sparsity_criteria)
            elif sparsity_criteria != -1 and snr_criteria != -1 :
                return (sparsity == sparsity_criteria) and (snr == snr_criteria)
            return True

        tp_results = [] # List of int of true_positives metrics
        # Iterate over the result_dict of each signal in the sparvar_results
        for result_dict in sparvar_results :
            signal_id = result_dict['id']
            signal_dict = self.signalDictFromId(signal_id)
            signal_sparsity, signal_snr = len(signal_dict['atoms']), signal_dict['snr']
            # Check if the signal satisfies the criteria
            if signal_condition(signal_sparsity, signal_snr) :
                true_atoms = signal_dict['atoms']
                approx_atoms = result_dict['atoms']
                tp = self.computeMaxTruePositives(true_atoms, approx_atoms, pos_err_threshold, corr_err_threshold)
                tp_results.append(tp)

        return tp_results

    def criticalDifferenceDiagramFromDB(self, **kwargs) :
        """
        Plot the critical difference diagram from the results of any sparVar database. 
        """
        pos_err_threshold = 10
        corr_err_threshold = 0.75
        sparvar_db_paths = {}
        verbose = False
        snr_criteria = -1
        sparsity_criteria = -1
        file_title = "example"

        def remove_digit(input_string):
            result = ""
            for char in input_string:
                if not char.isdigit():
                    result += char
            return result

        for key, value in kwargs.items() :
            if key == 'pos_err_threshold' :
                pos_err_threshold = value
            elif key == 'corr_err_threshold' :
                corr_err_threshold = value
            elif key == 'sparsity_criteria' :
                sparsity_criteria = value
            elif key == 'snr_criteria' :
                snr_criteria = value
            elif key == 'verbose' :
                verbose = value
            elif key == 'file_title' :
                file_title = value
            else :
                sparvar_db_paths[str(key).lower()] = value

        tp_values_per_algo = {}

        for i, (algorithm, path) in enumerate(sparvar_db_paths.items()) :
            if verbose :
                    print(f'\nProcessing {algorithm.upper()} results from {path}')
            if algorithm != 'l1' :
                results_key = remove_digit(algorithm)
                tp_values = self.computeCDCTruePositivesFromDB(
                    sparvar_db_path = path,
                    results_key = results_key,
                    pos_err_threshold = pos_err_threshold,
                    corr_err_threshold = corr_err_threshold,
                    sparsity_criteria = sparsity_criteria,
                    snr_criteria = snr_criteria,
                    verbose=verbose
                    )
                label = f'conv-{algorithm.upper()}'
                tp_values_per_algo[label] = tp_values
            else :
                tp_values = self.computeCDCTruePositivesFromAlphaCSC(
                    alphaCSC_db_path = path,
                    results_key = 'results',
                    pos_err_threshold = pos_err_threshold,
                    corr_err_threshold = corr_err_threshold,
                    sparsity_criteria = sparsity_criteria,
                    snr_criteria = snr_criteria,
                    verbose=verbose
                    )
                label = f'CSC-{algorithm.upper()}'
                tp_values_per_algo[label] = tp_values

        # Create a DataFrame from the dictionary
        df = pd.DataFrame(tp_values_per_algo)

        # create a CD diagram from the Pandas DataFrame
        diagram = Diagram(
            df.to_numpy(),
            treatment_names = df.columns,
            maximize_outcome = True
        )

        diagram.average_ranks # the average rank of each treatment
        diagram.get_groups(alpha=.05, adjustment="holm")

        diag_title = "CSC algorithms critical difference diagram"
        if snr_criteria != -1 :
            diag_title += f" + snr={snr_criteria}"
        if sparsity_criteria != -1 :
            diag_title += f" + spar={sparsity_criteria}"

        # export the diagram to a file
        diagram.to_file(
            f"{file_title}.pdf",
            alpha = .05,
            adjustment = "holm",
            reverse_x = True,
            axis_options = {"title": diag_title},
        )
        