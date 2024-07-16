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
from scipy.interpolate import interp1d

from .dictionary import ZSDictionary
from .atoms import ZSAtom
from .mmp import MMPTree
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
            approx = np.zeros_like(signal_dict['signal'])
            atoms_dict = result_dict['atoms']
            for atom_dict in atoms_dict :
                zs_atom = ZSAtom(atom_dict['b'], atom_dict['y'], atom_dict['s'])
                zs_atom.padBothSides(self.dictionary.getAtomsLength())
                atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom_dict['x'])
                approx += atom_signal
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
            approx = np.zeros_like(signal_dict['signal'])
            atoms_dict = result_dict['atoms']
            for atom_dict in atoms_dict :
                zs_atom = ZSAtom(atom_dict['b'], atom_dict['y'], atom_dict['s'])
                zs_atom.padBothSides(self.dictionary.getAtomsLength())
                atom_signal = zs_atom.getAtomInSignal(len(signal_dict['signal']), atom_dict['x'])
                approx += atom_signal
            axs[i+1].plot(signal_dict['signal'], label='Noisy signal', color='k', alpha=0.4, lw=3)
            axs[i+1].plot(true_signal, color='g', label='True signal')
            axs[0].plot(approx, color=f'C{i}', label=results_name[i])
            axs[i+1].plot(approx, color=f'C{i}', label=results_name[i])
            axs[i+1].set_title('       {} : MSE = {:.2e}'.format(results_name[i], result_dict["mse"]), fontsize=14, loc='left')
            axs[i+1].legend(loc='best')
            axs[i+1].axis('off')  
        axs[0].set_title('       True signal', fontsize=14, loc='left')
        axs[0].legend(loc='best') 
        axs[0].axis('off')

        # Add a colorbar
        cbar = fig.colorbar(sm, ax=axs, orientation='vertical', label='Overlap level', pad=0.01)
        cbar.solids.set(alpha=0.4)

        plt.show()

#    8888888b.   .d88888b.   .d8888b.  
#    888   Y88b d88P" "Y88b d88P  Y88b 
#    888    888 888     888 888    888 
#    888   d88P 888     888 888        
#    8888888P"  888     888 888        
#    888 T88b   888     888 888    888 
#    888  T88b  Y88b. .d88P Y88b  d88P 
#    888   T88b  "Y88888P"   "Y8888P"  


    def computeROCMetricsPosition(self, true_atoms:List[Dict], predicted_atoms:List[Dict]) -> Tuple:
        """
        Compute the ROC metrics for the given true and predicted atoms.
        """
        # Get the number of true and predicted atoms
        nb_true_atoms = len(true_atoms)
        nb_approx_atoms = len(predicted_atoms)

        # Atom matching
        matched_atoms = self.computeMatchingPosition(true_atoms, predicted_atoms)

        # Compute the True Positive, False Positive and False Negative
        position_errors = [abs(true_atom['x'] - predicted_atom['x']) for true_atom, predicted_atom in matched_atoms]
        tp = sum(1 for error in position_errors if error <= 5)
        fp = nb_approx_atoms - tp  
        fn = nb_true_atoms - tp       

        return tp, fp, fn, nb_true_atoms, nb_approx_atoms

    def calculateROCMetrics(self, row:pd.Series) -> Tuple:
        """
        Calculate the ROC metrics for the given row and position
        Returns:
            pd.Series: (TPR, FPR)
        """
        true_atoms = row['true_atoms']
        predicted_atoms = row['predicted_atoms']

        tp, fp, fn, nb_true_atoms, nb_approx_atoms = self.computeROCMetricsPosition(true_atoms, predicted_atoms)

        return pd.Series([tp, fp, fn, nb_true_atoms, nb_approx_atoms])

    def computeROCDataframe(self, db_path:str, process_data_func) :
        """ 
        Return the ROC dataframes for the given MP database.
        """
        processed_results = process_data_func(db_path)
        # Convert data to DataFrame
        df = pd.DataFrame(processed_results['results'])

        # Apply the metrics function
        metrics = df.apply(self.calculateROCMetrics, axis=1)
        df[['tp', 'fp', 'fn', 'nb_true_atoms', 'nb_predicted_atoms']] = metrics

        # Compute the FPR
        df['fpr'] = np.where((df['nb_true_atoms'] - df['tp'] + df['fp']) > 0, df['fp'] / (df['nb_true_atoms'] - df['tp'] + df['fp']), 0)

        # Compute the TPR
        df['tpr'] = np.where((df['tp'] + df['fn']) > 0, df['tp'] / (df['tp'] + df['fn']), 0)

        return df
        
    def plotROC(self, mmpdf_db_path: str, mp_db_path: str) -> None :
        """
        Plot the F1 score by SNR for different algorithms and sparsity levels.
        """
        plt.figure(figsize=(12, 8))

        # MMP-DF database
        mmpdf_metrics_df = self.computeROCDataframe(db_path=mmpdf_db_path, process_data_func=self.processMMPDFResults)

        # MP database
        mp_metrics_df = self.computeROCDataframe(db_path=mp_db_path, process_data_func=self.processMPResults)

        # Merge the dataframes
        metrics_df = pd.concat([mmpdf_metrics_df, mp_metrics_df], ignore_index=True)

        # Define colors and markers for the plots
        colors = {'OMP': 'navy', 'MMP-DF': 'red', 'MP': 'green'}
        markers = {3: 'o', 4: 'D', 5: 'X'}  # Example for up to 5 sparsity levelsprint(metrics_df.loc[metrics_df['algo_type'] == 'MMP-DF', 'fpr'].unique())

        print(f"MMP-DF FPR : {metrics_df.loc[metrics_df['algo_type'] == 'MMP-DF', 'fpr'].unique()}")
        print(f"MMP-DF TPR : {metrics_df.loc[metrics_df['algo_type'] == 'MMP-DF', 'tpr'].unique()}\n")
        print(f"OMP FPR : {metrics_df.loc[metrics_df['algo_type'] == 'OMP', 'fpr'].unique()}")
        print(f"OMP TPR : {metrics_df.loc[metrics_df['algo_type'] == 'OMP', 'tpr'].unique()}\n")
        print(f"MP FPR : {metrics_df.loc[metrics_df['algo_type'] == 'MP', 'fpr'].unique()}")
        print(f"MP TPR : {metrics_df.loc[metrics_df['algo_type'] == 'MP', 'tpr'].unique()}\n")

        # Group data and plot
        sns.lineplot(x='fpr', y='tpr', hue='algo_type', data=metrics_df, palette="flare")
        sns.despine(trim=True)

        plt.title(f'Receiver Operating Characteristic', fontsize=16)
        plt.xlabel('False positive rate', fontsize=14)
        plt.ylabel('True positive rate', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Algorithm', loc='best', fontsize=12)
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
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
    
    def computeTPFPFNMetrics(self, true_atoms, approx_atoms, sparsity, position_error_threshold:int=20, verbose:bool=False) -> Tuple:
        """
        Compute the True Positive, False Positive and False Negative for the given true and approx atoms.
        """
        # Get the number of true and approx atoms
        
        nb_approx_atoms = len(approx_atoms)
        nb_true_atoms = len(true_atoms)
        #true_atoms = true_atoms[:nb_approx_atoms]

        # Atom matching: Hungarian matching ensuring len(matched_atoms) = nb_true_atoms
        matched_atoms = self.computeMatchingPosition(true_atoms, approx_atoms)

        # Compute the True Positive, False Positive and False Negative
        position_errors = [abs(true_atom['x'] - approx_atom['x']) for true_atom, approx_atom in matched_atoms]
        tp = sum(1 for error in position_errors if error <= position_error_threshold)

        # False Positives
        fp = len(approx_atoms) - tp  
        fp += sparsity - len(approx_atoms)

        # False Negatives
        fn = nb_true_atoms - tp

        if verbose :
            print(f'Sparsity = {sparsity} | Position Error Threshold = {position_error_threshold}')
            for match in matched_atoms :
                print(f'    t : {match[0]["x"]}  |  a : {match[1]["x"]}  ==> {bool(abs(match[0]["x"] - match[1]["x"]) <= position_error_threshold)}')
            print(f'    ==> TP : {tp}  |  FP : {fp}  |  FN : {fn} \n')
            print('\n')

        return tp, fp, fn

    def computePrecisionRecallMetrics(self, true_atoms, approx_atoms, sparsity, position_error_threshold:int=20, verbose:bool=False) -> Tuple:
        """
        Compute the precision-recall metrics for the given true and approx atoms.
        """
        tp, fp, fn = self.computeTPFPFNMetrics(true_atoms, approx_atoms, sparsity, position_error_threshold=position_error_threshold, verbose=verbose)

        # Calculate Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return precision, recall
    
    def computePrecisionRecallMetrics(self, true_atoms, approx_atoms, sparsity, position_error_threshold:int=20, verbose:bool=False) -> Tuple:
        """
        Compute the precision-recall metrics for the given true and approx atoms.
        """
        # Get the number of true and approx atoms
        nb_approx_atoms = len(approx_atoms)
        nb_true_atoms = len(true_atoms)

        # Atom matching: Hungarian matching ensuring len(matched_atoms) = nb_true_atoms
        matched_atoms = self.computeMatchingPosition(true_atoms, approx_atoms)

        # Compute the True Positive, False Positive and False Negative
        position_errors = [abs(true_atom['x'] - approx_atom['x']) for true_atom, approx_atom in matched_atoms]
        tp = sum(1 for error in position_errors if error <= position_error_threshold)

        precision = tp / sparsity
        recall = tp / nb_true_atoms

        return precision, recall

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

    def extractPRCurveData_MMPDF(self, mmpdf_dict:dict, max_branches:int=10, max_sparsity:int=10, verbose:bool=False) -> pd.DataFrame:
        """
        Process the MMP-DF results.
        Args :
            mmpdf_dict (dict) : The MMP-DF dictionary.
            signal (np.ndarray) : The signal.
            max_branches (int) : The maximum number of branches to consider.
            max_sparsity (int) : The maximum sparsity level to consider.
        Returns :
            pd.DataFrame : The precision-recall dataframe.
        """
        # Retrieve the tree structure to the max number of branches
        mmp_tree_dict = MMPTree.shrinkMMPTreeDict(mmpdf_dict['mmp-tree'], max_branches)
        
        # Retrieve the signal from the datas
        signal_dict = self.signalDictFromId(mmpdf_dict['id'])
        signal = signal_dict['signal']
        true_atoms = signal_dict['atoms']

        # Initialize the precion-recall dataframe
        precisions = []
        recalls = []

        # Iterate over the attended sparsity levels
        sparsity_levels = [i+1 for i in range(max_sparsity)]
        for candidate_sparsity in sparsity_levels :
            candidate_atoms = MMPTree.mmpdfCandidateFromMMPTreeDict(mmp_tree_dict, self.dictionary.getAtomsLength(), signal, candidate_sparsity=candidate_sparsity)
            precision, recall = self.computePrecisionRecallMetrics(true_atoms, candidate_atoms, candidate_sparsity, position_error_threshold=20, verbose=verbose)
            precisions.append(precision)
            recalls.append(recall)

        precision_recall_df = pd.DataFrame({'sparsity': sparsity_levels, 'precision': precisions, 'recall': recalls})
        return precision_recall_df
    
    def extractPRCurveData_OMP(self, mmpdf_dict:dict, max_branches:int=10, max_sparsity:int=10, verbose:bool=False) -> pd.DataFrame:
        """
        Process the MMP-DF results.
        Args :
            mmpdf_dict (dict) : The MMP-DF dictionary.
            signal (np.ndarray) : The signal.
            max_branches (int) : The maximum number of branches to consider.
            max_sparsity (int) : The maximum sparsity level to consider.
        Returns :
            pd.DataFrame : The precision-recall dataframe.
        """
        # Retrieve the tree structure to the max number of branches
        mmp_tree_dict = MMPTree.shrinkMMPTreeDict(mmpdf_dict['mmp-tree'], max_branches)
        
        # Retrieve the signal from the datas
        signal_dict = self.signalDictFromId(mmpdf_dict['id'])
        signal = signal_dict['signal']
        true_atoms = signal_dict['atoms']

        # Initialize the precion-recall dataframe
        precisions = []
        recalls = []

        # Iterate over the attended sparsity levels
        sparsity_levels = [i+1 for i in range(max_sparsity)]
        for candidate_sparsity in sparsity_levels :
            candidate_atoms = MMPTree.ompCandidateFromMMPTreeDict(mmp_tree_dict, signal, candidate_sparsity=candidate_sparsity)
            precision, recall = self.computePrecisionRecallMetrics(true_atoms, candidate_atoms, candidate_sparsity, position_error_threshold=20, verbose=verbose)
            precisions.append(precision)
            recalls.append(recall)

        precision_recall_df = pd.DataFrame({'sparsity': sparsity_levels, 'precision': precisions, 'recall': recalls})
        return precision_recall_df

    def extractPRCurveData_MP(self, mp_dict:dict, max_branches:int=10, max_sparsity:int=10, verbose:bool=False) -> pd.DataFrame:
        """
        Process the MMP-DF results.
        Args :
            mp_dict (dict) : The MMP-DF dictionary.
            signal (np.ndarray) : The signal.
            max_branches (int) : The maximum number of branches to consider.
            max_sparsity (int) : The maximum sparsity level to consider.
        Returns :
            pd.DataFrame : The precision-recall dataframe.
        """
        # Retrieve the atoms from the MP algorithm output
        approx_atoms = mp_dict['atoms']

        # Retrieve the signal from the datas
        signal_dict = self.signalDictFromId(mp_dict['id'])
        signal = signal_dict['signal']
        true_atoms = signal_dict['atoms']

        # Initialize the precion-recall dataframe
        precisions = []
        recalls = []

        # Iterate over the attended sparsity levels
        sparsity_levels = [i+1 for i in range(max_sparsity)]
        for candidate_sparsity in sparsity_levels :
            candidate_atoms = approx_atoms[:min(candidate_sparsity, len(approx_atoms))]
            precision, recall = self.computePrecisionRecallMetrics(true_atoms, candidate_atoms, candidate_sparsity, position_error_threshold=20, verbose=verbose)
            precisions.append(precision)
            recalls.append(recall)

        precision_recall_df = pd.DataFrame({'sparsity': sparsity_levels, 'precision': precisions, 'recall': recalls})
        return precision_recall_df

    def displayPRCurve_MMPDF(self, mmpdf_dict: dict, max_branches: int = 10, max_sparsity: int = 10, verbose:bool=False) :
        """
        Display the precision-recall curve for the given MMP-DF results.
        
        Args:
            mmpdf_dict (dict): The MMP-DF dictionary.
            max_branches (int): The maximum number of branches to consider.
            max_sparsity (int): The maximum sparsity level to consider.
        """
        # Step 1: Extract precision-recall data as DataFrame
        pr_df = self.extractPRCurveData_MMPDF(mmpdf_dict, max_branches, max_sparsity, verbose=verbose)

        # Step 2: Convert DataFrame to numpy array
        pr_array = pr_df[['precision', 'recall']].values

        # Step 3: Plot the precision-recall curve
        ax = CSCWorkbench.plotPRCurve(pr_array)
        plt.show()

    def displayPRCurveFromId_MMPDF(self, mmpdf_db_path:str, id:int, max_branches:int=10, max_sparsity:int=10, verbose:bool=False) :
        # Load the data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)
            mmpdf_dict = next((result for result in output_data['mmp'] if result['id'] == id), None)

        self.displayPRCurve_MMPDF(mmpdf_dict, max_branches=max_branches, max_sparsity=max_sparsity, verbose=verbose)

    def displayPRCurveComparisonFromId(self, mmpdf_db_path:str, mp_db_path:str, id:int, max_branches:int=10, max_sparsity:int=10, verbose:bool=False) :
        # Load the MMP-DF data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)
            mmpdf_dict = next((result for result in output_data['mmp'] if result['id'] == id), None)
        
        # Load the MP data
        with open(mp_db_path, 'r') as f:
            output_data = json.load(f)
            mp_dict = next((result for result in output_data['mp'] if result['id'] == id), None)

        # Step 1: Extract precision-recall data as DataFrame
        mmpdf_pr_df = self.extractPRCurveData_MMPDF(mmpdf_dict, max_branches, max_sparsity, verbose=verbose)
        omp_pr_df = self.extractPRCurveData_OMP(mmpdf_dict, max_branches, max_sparsity, verbose=verbose)
        mp_pr_df = self.extractPRCurveData_MP(mp_dict, max_branches, max_sparsity, verbose=verbose)
        
        # Step 2: Convert DataFrame to numpy array
        mmpdf_pr_array = mmpdf_pr_df[['precision', 'recall']].values
        omp_pr_array = omp_pr_df[['precision', 'recall']].values
        mp_pr_array = mp_pr_df[['precision', 'recall']].values

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        CSCWorkbench.plotPRCurve(mmpdf_pr_array, ax=ax, label='MMP-DF')
        CSCWorkbench.plotPRCurve(omp_pr_array, ax=ax, label='OMP')
        CSCWorkbench.plotPRCurve(mp_pr_array, ax=ax, label='MP')
        plt.title(f'Precision-Recall Curve for signal n°{id}')
        plt.legend(loc='best')
        plt.show()

    def displayPRCDecomposition(self, mmpdf_db_path:str, id:int, verbose:bool=False) :
        # Load the data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)
            mmpdf_dict = next((result for result in output_data['mmp'] if result['id'] == id), None)

        max_branches = 20
        max_sparsity = 20

        # Step 1: Extract precision-recall data as DataFrame
        pr_df = self.extractPRCurveData_MMPDF(mmpdf_dict, max_branches, max_sparsity)

        # Retrieve the tree structure to the max number of branches
        mmp_tree_dict = MMPTree.shrinkMMPTreeDict(mmpdf_dict['mmp-tree'], max_branches)
        
        # Retrieve the signal from the datas
        signal_dict = self.signalDictFromId(mmpdf_dict['id'])
        signal = signal_dict['signal']
        true_atoms = signal_dict['atoms']

        # Iterate over the attended sparsity levels
        sparsity_levels = [i+1 for i in range(max_sparsity)]

        fig, axs = plt.subplots(len(sparsity_levels), 2, figsize=(15, 4*len(sparsity_levels)),
                            gridspec_kw={'width_ratios': [3, 1]}, sharex='col')
        
        for i, candidate_sparsity in enumerate(sparsity_levels) :

            # Extract the candidate atoms
            candidate_atoms = MMPTree.mmpdfCandidateFromMMPTreeDict(mmp_tree_dict, self.dictionary.getAtomsLength(), signal, candidate_sparsity=candidate_sparsity)

            # Plot the noisy signal
            axs[i, 0].plot(signal, label='Noisy signal', color='k', alpha=0.4, lw=3)

            matched_atoms = self.computeMatchingPosition(true_atoms, candidate_atoms)
            
            alphas = [0.4]*len(matched_atoms)
            alphas[-1] = 1

            for j, (true_atom, cand_atom) in enumerate(matched_atoms) : 
                # Plot the true atoms
                t_atom = ZSAtom.from_dict(true_atom)
                t_atom.padBothSides(self.dictionary.getAtomsLength())
                t_atom_signal = t_atom.getAtomInSignal(len(signal), true_atom['x'])
                axs[i, 0].plot(t_atom_signal, label=f'True atom n°{j+1} at x={true_atom["x"]}', lw=1, color='g', alpha=alphas[j])
                # Plot the cand atoms
                c_atom = ZSAtom.from_dict(cand_atom)
                c_atom.padBothSides(self.dictionary.getAtomsLength())
                c_atom_signal = c_atom.getAtomInSignal(len(signal), cand_atom['x'])
                axs[i, 0].plot(c_atom_signal, label=f'Cand atom n°{j+1} at x={cand_atom["x"]}', lw=1, color='b', alpha=alphas[j])

            # Compute the precision and the recall with the TP, FP, FN Metrics
            tp, fp, fn = self.computeTPFPFNMetrics(true_atoms, candidate_atoms, candidate_sparsity, position_error_threshold=20, verbose=verbose)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            axs[i, 0].set_title(f'Step n°{i} : sparsity {candidate_sparsity} ==> Number of atoms : {len(matched_atoms)}', fontsize=14)
            axs[i, 0].legend(loc='best')
            axs[i, 0].axis('off')

            # Displaying TP, FP, FN in a table on the right axis
            colLabels = ["TP", "FP", "FN"]
            rowLabels = ["Value"]
            table_data = [[tp, fp, fn]]
            table = axs[i, 1].table(cellText=table_data, colLabels=colLabels, rowLabels=rowLabels, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            axs[i, 1].set_title(f'Precision = {round(precision, 2)} ; Recall = {round(recall, 2)}', fontsize=14)
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def computePRCurvesFromSparsity(self, mmpdf_db_path:str, sparsity:int, verbose:bool=False) :
        # Load the data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)
            mmp_results = output_data['mmp']

        max_branches = 10
        max_sparsity = 10

        all_pr = []

        for mmp_dict in mmp_results :
            if mmp_dict['sparsity'] == sparsity :
                pr_df = self.extractPRCurveData_MMPDF(mmp_dict, max_branches, max_sparsity, verbose=verbose)
                pr_array = pr_df[['precision', 'recall']].values
                all_pr.append(pr_array)

        return all_pr

    def displayMeanPRCurveFromSparsity(self, mmpdf_db_path:str, sparsity:int, verbose:bool=False) :
        
        all_pr = self.computePRCurvesFromSparsity(mmpdf_db_path, sparsity)

        fig, ax = plt.subplots()
        pr_mean, pr_mean_plus_std, pr_mean_minus_std = self.computeMeanPRCurve(all_pr, n_samples=1000, verbose=verbose)

        CSCWorkbench.plotPRCurve(pr_mean, ax=ax, color="k", alpha=1)
        CSCWorkbench.plotPRCurve(pr_mean_plus_std, ax=ax, color="r", alpha=0.5)
        CSCWorkbench.plotPRCurve(pr_mean_minus_std, ax=ax, color="r", alpha=0.5)

    def computePRCurves_MMPDF(self, mmpdf_db_path:str, max_branches:int, max_sparsity:int, verbose:bool=False) :
        # Load the data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)
            mmp_results = output_data['mmp']

            all_pr_dict = {}

            for sparsity in output_data['sparsityLevels'] :
                all_pr_dict[sparsity] = list()

        for mmp_dict in mmp_results :
            pr_df = self.extractPRCurveData_MMPDF(mmp_dict, max_branches, max_sparsity, verbose=verbose)
            pr_array = pr_df[['precision', 'recall']].values
            all_pr_dict[mmp_dict['sparsity']].append(pr_array)

        return all_pr_dict
    
    def computePRCurves_OMP(self, mmpdf_db_path:str, max_branches:int, max_sparsity:int, verbose:bool=False) :
        # Load the data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)
            mmp_results = output_data['mmp']

            all_pr_dict = {}

            for sparsity in output_data['sparsityLevels'] :
                all_pr_dict[sparsity] = list()


        for mmp_dict in mmp_results :
            pr_df = self.extractPRCurveData_OMP(mmp_dict, max_branches, max_sparsity, verbose=verbose)
            pr_array = pr_df[['precision', 'recall']].values
            all_pr_dict[mmp_dict['sparsity']].append(pr_array)

        return all_pr_dict

    def computePRCurves_MP(self, mp_db_path:str, max_branches:int, max_sparsity:int, verbose:bool=False) :
        # Load the data
        with open(mp_db_path, 'r') as f:
            output_data = json.load(f)
            mp_results = output_data['mp']

            all_pr_dict = {}

            for sparsity in output_data['sparsityLevels'] :
                all_pr_dict[sparsity] = list()

        for mp_dict in mp_results :
            pr_df = self.extractPRCurveData_MP(mp_dict, max_branches, max_sparsity, verbose=verbose)
            pr_array = pr_df[['precision', 'recall']].values
            try :
                all_pr_dict[mp_dict['sparsity']].append(pr_array)
    
            except KeyError :
                # Unexplained error in csc-mp-200.json
                # "id": 2165
                # "sparsity": 2
                pass

        return all_pr_dict
    
    def computeMeanPRCComparisonFromId(self, mmpdf_db_path:str, mp_db_path:str, id:int, max_branches:int=10, max_sparsity:int=10, verbose:bool=False) :

        # Load the MMP-DF data
        with open(mmpdf_db_path, 'r') as f:
            output_data = json.load(f)
            mmpdf_dict = next((result for result in output_data['mmp'] if result['id'] == id), None)
        
        # Load the MP data
        with open(mp_db_path, 'r') as f:
            output_data = json.load(f)
            mp_dict = next((result for result in output_data['mp'] if result['id'] == id), None)

        # Step 1: Extract precision-recall data as DataFrame
        mmpdf_pr_df = self.extractPRCurveData_MMPDF(mmpdf_dict, max_branches, max_sparsity, verbose=verbose)
        omp_pr_df = self.extractPRCurveData_OMP(mmpdf_dict, max_branches, max_sparsity, verbose=verbose)
        mp_pr_df = self.extractPRCurveData_MP(mp_dict, max_branches, max_sparsity, verbose=verbose)
        
        # Step 2: Convert DataFrame to numpy array
        mmpdf_pr_array = mmpdf_pr_df[['precision', 'recall']].values
        omp_pr_array = omp_pr_df[['precision', 'recall']].values
        mp_pr_array = mp_pr_df[['precision', 'recall']].values

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        CSCWorkbench.plotPRCurve(mmpdf_pr_array, ax=ax, label='MMP-DF')
        CSCWorkbench.plotPRCurve(omp_pr_array, ax=ax, label='OMP')
        CSCWorkbench.plotPRCurve(mp_pr_array, ax=ax, label='MP')
        plt.title(f'Precision-Recall Curve for signal n°{id}')
        plt.legend(loc='best')
        plt.show()

    def displayMeanPRC(self, mmpdf_db_path:str, mp_db_path:str, max_branches:int=10, max_sparsity:int=50, verbose:bool=False) :

        mmpdf_all_pr_dict = self.computePRCurves_MMPDF(mmpdf_db_path, max_branches=max_branches, max_sparsity=max_sparsity)
        omp_all_pr_dict = self.computePRCurves_OMP(mmpdf_db_path, max_branches=10, max_sparsity=max_sparsity)

        mmpdf_prc = list()
        omp_prc = list()

        # MMP-DF
        for sparsity, arrays in mmpdf_all_pr_dict.items():
            if verbose :
                print(f'MMP-DF: sparsity={sparsity} : {len(arrays)} x {arrays[0].shape}')
            for signal_prc in arrays :
                mmpdf_prc.append(signal_prc)

        # OMP
        for sparsity, arrays in omp_all_pr_dict.items():
            if verbose :
                print(f'MMP-DF: sparsity={sparsity} : {len(arrays)} x {arrays[0].shape}')
            for signal_prc in arrays :
                omp_prc.append(signal_prc)

        mmpdf_pr_mean, mmpdf_pr_mean_plus_std, mmpdf_pr_mean_minus_std = self.computeMeanPRCurve(mmpdf_prc, n_samples=1000)
        omp_pr_mean, omp_pr_mean_plus_std, omp_pr_mean_minus_std = self.computeMeanPRCurve(omp_prc, n_samples=1000)

        fig, ax = plt.subplots()

        # Plot mean MMP-DF curve
        CSCWorkbench.plotPRCurve(mmpdf_pr_mean, ax=ax, color="C1", label="MMP-DF", alpha=1)
        CSCWorkbench.plotPRCurve(mmpdf_pr_mean_plus_std, ax=ax, color="C1", alpha=0.5)
        CSCWorkbench.plotPRCurve(mmpdf_pr_mean_minus_std, ax=ax, color="C1", alpha=0.5)

        # Plot mean OMP curve
        CSCWorkbench.plotPRCurve(omp_pr_mean, ax=ax, color="b", label="OMP", alpha=1)
        CSCWorkbench.plotPRCurve(omp_pr_mean_plus_std, ax=ax, color="b", alpha=0.5)
        CSCWorkbench.plotPRCurve(omp_pr_mean_minus_std, ax=ax, color="b", alpha=0.5)

        plt.title('Precision-Recall curves for OMP and MMP-DF')
        plt.legend(loc='best')
        plt.show()

    def displayMeanPRCFromDict(self, mmpdf_all_pr_dict:dict, omp_all_pr_dict:dict, max_branches:int=10, max_sparsity:int=50, verbose:bool=False) :

        mmpdf_prc = list()
        omp_prc = list()

        # MMP-DF
        for sparsity, arrays in mmpdf_all_pr_dict.items():
            for signal_prc in arrays :
                mmpdf_prc.append(signal_prc)

        # OMP
        for sparsity, arrays in omp_all_pr_dict.items():
            for signal_prc in arrays :
                omp_prc.append(signal_prc)

        mmpdf_pr_mean, mmpdf_pr_mean_plus_std, mmpdf_pr_mean_minus_std = self.computeMeanPRCurve(mmpdf_prc, n_samples=1000)
        omp_pr_mean, omp_pr_mean_plus_std, omp_pr_mean_minus_std = self.computeMeanPRCurve(omp_prc, n_samples=1000)

        fig, ax = plt.subplots()

        # Plot mean MMP-DF curve
        CSCWorkbench.plotPRCurve(mmpdf_pr_mean, ax=ax, color="C1", label="MMP-DF", alpha=1)
        CSCWorkbench.plotPRCurve(mmpdf_pr_mean_plus_std, ax=ax, color="C1", alpha=0.5)
        CSCWorkbench.plotPRCurve(mmpdf_pr_mean_minus_std, ax=ax, color="C1", alpha=0.5)

        # Plot mean OMP curve
        CSCWorkbench.plotPRCurve(omp_pr_mean, ax=ax, color="b", label="OMP", alpha=1)
        CSCWorkbench.plotPRCurve(omp_pr_mean_plus_std, ax=ax, color="b", alpha=0.5)
        CSCWorkbench.plotPRCurve(omp_pr_mean_minus_std, ax=ax, color="b", alpha=0.5)

        plt.title('Precision-Recall curves for OMP and MMP-DF')
        plt.legend(loc='best')
        plt.show()
