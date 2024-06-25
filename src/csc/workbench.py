import os
import time
import json
import itertools
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List, Dict, Tuple, Any, Union
from scipy import optimize

from .dictionary import ZSDictionary
from .atoms import ZSAtom
from .utils import *

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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
        true_positions = np.array([atom['x'] for atom in true_atoms_dict])
        approx_positions = np.array([atom['x'] for atom in approx_atoms_dict])

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
    
    def mseErrorThresholdAutoCalibration(self, error_prct=10) -> None :
        """
        Auto-calibration of the MSE error threshold.
        """
        assert 0 <= error_prct <= 100, "Error percentage must be between 0 and 100."
        self.mseErrorThreshold = {}
        data = self.loadDataFromPath(self.signals_path)

        for snr in self.snr_levels :
            signal_dict = next((item for item in data['signals'] if item['snr'] == snr), None)
            signal_atoms = signal_dict['atoms']
            recovered_signal = np.zeros_like(signal_dict['signal'])
            for atom in signal_atoms:
                atom_obj = ZSAtom.from_dict(atom)
                atom_obj.padBothSides(self.dictionary.getAtomsLength())
                recovered_signal += atom_obj.getAtomInSignal(len(recovered_signal), atom['x'])
            mse = np.mean((signal_dict['signal'] - recovered_signal)**2)
            self.mseErrorThreshold[snr] = mse*(1 + error_prct/100)

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
        fig, axs = plt.subplots(len(ids), 1, figsize=(12, 3*len(ids)), sharex=True)
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
            axs[i].axis('off')
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
            axs[i+1].axis('off')

        axs[0].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
        axs[0].plot(trueSuperposition, label='True superposition', color='g', alpha=0.9, lw=2)
        axs[0].plot(approxSuperposition, label='Approx superposition')
        axs[0].set_title('Signal n°{} decomposition : MSE = {:.2e}'.format(id, approx_dict['mse']))
        axs[0].legend(loc='best')
        axs[0].axis('off')
        plt.show()


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
    
    def plotMMPTree(self, db_path:str, id:int) -> None :

        """
        Plot the MMP Tree leaves' approximations.
        """
        # Load the data
        with open(db_path, 'r') as f:
            output_data = json.load(f)
            mmp_result_dict = next((result for result in output_data['mmp'] if result['id'] == id), None)
        # Get the true signal
        signal_dict = self.signalDictFromId(id)
        mmp_tree_dict = mmp_result_dict['mmp-tree']

        # Plot the comparison
        fig, axs = plt.subplots(len(mmp_tree_dict)+1, 1, figsize=(12, 2*(len(mmp_tree_dict)+1)), sharex=True)
        axs[0].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
            
        for i, (path_str, path_dict) in enumerate(mmp_tree_dict.items()) :
            atoms_dict = path_dict['atoms']
            mmp_approx = np.zeros_like(signal_dict['signal'])
            for atom_dict in atoms_dict :
                zs_atom = ZSAtom(atom_dict['b'], atom_dict['y'], atom_dict['sigma'])
                zs_atom.padBothSides(self.dictionary.getAtomsLength())
                atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom_dict['x'])
                mmp_approx += atom_signal
            axs[0].plot(mmp_approx, label='{} = {}'.format(path_str, path_dict['mse']))
            axs[i+1].plot(mmp_approx, label='{} = {}'.format(path_str, path_dict['mse']))
            axs[i+1].legend(loc='best')
            axs[i+1].axis('off')

        axs[0].legend(loc='best')
        axs[0].axis('off')
        plt.show()

    def plotMMPComparison(self, db_path:str, id:int) -> None :
        """
        Use three subplots to compare the results between the OMP and the MMP results.
        The OMP result corresponds to the first branch of the MMP tree.
        The MMP result corresponds to the MSE-argmin of the MMP tree.
        """
        # Load the data
        with open(db_path, 'r') as f:
            output_data = json.load(f)
            mmp_result_dict = next((result for result in output_data['mmp'] if result['id'] == id), None)
        # Get the true signal
        signal_dict = self.signalDictFromId(id)
        mmp_tree_dict = mmp_result_dict['mmp-tree']
        sparsity = mmp_result_dict['sparsity']

        fig, axs = plt.subplots(3, 1, figsize=(12, 3*3), sharex=True)
        axs[0].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
        
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

        # Extract the atoms from the dict
        results_dict = [omp_dict, mmp_dict]
        results_name = ['OMP', f'MMP {mmp_path}']
        # Plot the comparison
        for i, result_dict in enumerate(results_dict) :
            approx = np.zeros_like(signal_dict['signal'])
            atoms_dict = result_dict['atoms']
            for atom_dict in atoms_dict :
                zs_atom = ZSAtom(atom_dict['b'], atom_dict['y'], atom_dict['s'])
                zs_atom.padBothSides(self.dictionary.getAtomsLength())
                atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom_dict['x'])
                approx += atom_signal
            axs[0].plot(approx, color=f'C{i}', label=results_name[i])
            axs[i+1].plot(approx, color=f'C{i}')
            axs[i+1].plot(signal_dict['signal'], color='k', alpha=0.4, lw=3)
            axs[i+1].set_title('{} : MSE = {:.2e}'.format(results_name[i], result_dict["mse"]), fontsize=12)
            axs[i+1].axis('off')  

        axs[0].legend(loc='best') 
        axs[0].axis('off')
        plt.show()

    def plotMMPDecomposition(self, db_path:str, id:int) -> None :
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
        nb_atoms = len(omp_atoms_dict)

        fig, axs = plt.subplots(nb_atoms+1, 1, figsize=(12, (nb_atoms+1)*3), sharex=True)
    
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
            # Plot the atoms and the noisy signal
            axs[i+1].plot(signal_dict['signal'], label='Signal', color='k', alpha=0.3, lw=3)
            axs[i+1].plot(omp_atom_signal, label='OMP atom')
            axs[i+1].plot(mmp_atom_signal, label='MMP atom')
            axs[i+1].set_title(f'Step n°{i+1}', fontsize=12)
            axs[i+1].legend(loc='best')
            axs[i+1].axis('off')
        
        axs[0].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
        axs[0].plot(omp_signal, label='OMP')
        axs[0].plot(mmp_signal, label='MMP')
        axs[0].set_title('OMP MSE = {:.2e} & MMP {} MSE = {:.2e}'.format(omp_dict['mse'], mmp_path, mmp_dict['mse']), fontsize=12)
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

    def plotMMPDFPosErrComparison(self, mmpdf_db_path:str) :
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

    def plotMMPDFPosErrAtSparsity(self, mmpdf_db_path:str, sparsity:int) :
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

    def plotMMPDFPosErrAtStep(self, mmpdf_db_path:str, step:int) :
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

        
        omp_mse_signal = np.zeros_like(signal_dict['signal'])
        mmp_mse_signal = np.zeros_like(signal_dict['signal'])
        for (start, end), val in zip(overlap_intervals, overlap_intervals_values):
            # Compute the mse on the interval
            omp_mse_on_interval = np.mean((signal_dict['signal'][start:end] - omp_signal[start:end])**2)
            mmp_mse_on_interval = np.mean((signal_dict['signal'][start:end] - mmp_signal[start:end])**2)
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

    def computeMMPDFMSEPerOverlapInterval(self, db_path:str) -> Dict:
        """
        Compute the position errors for all the signals.
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
                omp_mse_on_interval = np.mean((signal_dict['signal'][start:end] - omp_signal[start:end])**2)
                mmp_mse_on_interval = np.mean((signal_dict['signal'][start:end] - mmp_signal[start:end])**2)

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
        data_overlap_intervals = self.computeMMPDFMSEPerOverlapInterval(mmpdf_db_path)
        df_all_snr = pd.DataFrame(data_overlap_intervals)
        df = df_all_snr.loc[df_all_snr['snr'] == snr]
        sns.boxplot(x='overlap', y='local_mse', hue='algo_type', data=df, palette="flare", fliersize=2, whis=1.5, showfliers=False)
        sns.despine(trim=True)
        plt.title('OMP vs MMPDF local MSE Comparison by local overlap level', fontsize=14)
        plt.xlabel('Signal to Noise Ratio (SNR) in dB', fontsize=12)
        plt.ylabel('Local MSE', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Algorithm', loc='best')
        plt.show()

    

    
        

