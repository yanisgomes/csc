import os
import time
import json
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
from scipy.signal import oaconvolve

from joblib import Parallel, delayed

from .atoms import ZSAtom
from .mmp import MMPTree
from .utils import *

def time_decorator(func):
    def wrapper(*args, **kwargs):
        if args[0].timing_enabled:
            # Measure the execution time
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Store the execution time
            func_name = func.__name__
            if func_name not in args[0].performance_log:
                args[0].performance_log[func_name] = []
            args[0].performance_log[func_name].append(elapsed_time)
            # Print the execution time
            verbose = kwargs.get('verbose', False)
            if verbose :
                print(f"Temps d'exécution de {func_name}: {elapsed_time:.6f} secondes.")
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper

class ZSDictionary() :

    def __init__(self, atoms_list:List[ZSAtom]) :        
        # Build the atoms and pad them to the same length
        self.atoms = atoms_list
        self.atoms_length = max([len(atom) for atom in self.atoms])
        for atom in self.atoms:
            atom.padBothSides(self.atoms_length)

        assert all([len(atom) == self.atoms_length for atom in self.atoms]), "All atoms must have the same length"

        # Timing performance
        self.timing_enabled = False
        self.performance_log = {}  # {method_name: execution_time}

    @classmethod
    def from_values(cls, b_values, y_values, s_values) -> 'ZSDictionary':
        """
        Create a ZSDictionary object from the given range of values for b, y, and s.

        Args:
            b_tuple (Tuple[float, float, float]): A tuple representing the range of values for b (b_min, b_max, b_step).
            y_tuple (Tuple[float, float, float]): A tuple representing the range of values for y (y_min, y_max, y_step).
            s_tuple (Tuple[float, float, float]): A tuple representing the range of values for s (s_min, s_max, s_step).

        Returns:
            ZSDictionary: A ZSDictionary object containing ZSAtom objects generated from the combinations of b, y, and s values.

        """
        atoms = list()
        combinations = list(product(b_values, y_values, s_values))
        # Create ZSAtom objects for each combination
        atoms = [ZSAtom(b, y, s) for b, y, s in combinations]
        return cls(atoms)

    def __str__(self) -> str:
        return "ZSMeasurementMatrix object with {} different atoms".format(len(self))
    
    def __len__(self) -> int:
        return len(self.atoms)
    
    def __contains__(self, item) -> bool:
        if isinstance(item, tuple):
            for atom in self.atoms :
                if atom.params['b'] == item[0] and atom.params['y'] == item[1] and atom.params['sigma'] == item[2] :
                    return True
            return False
        elif isinstance(item, ZSAtom):
            params = (item.params['b'], item.params['y'], item.params['sigma'])
            return params in self
        else:
            raise TypeError("Unsupported type for 'in' operation")
        
    def enable_timing(self):
        self.timing_enabled = True

    def disable_timing(self):
        self.timing_enabled = False

    def getAtom(self, idx:int) -> ZSAtom:
        return self.atoms[idx]

    def getAtoms(self) -> List[ZSAtom] :
        return self.atoms
    
    def getAtomsLength(self) -> int :
        return self.atoms_length
    
    def getAtomFromParams(self, b, y, s, tolerance=1e-5):
        """
        Get an atom from the dictionary based on its parameters.

        Args:
            b, y, s (float): The parameters of the atom.
            tolerance (float, optional): The tolerance for the comparison. Defaults to 1e-5.

        Returns:
            ZSAtom: The atom with the given parameters. If no such atom exists, returns None.
        """
        for atom in self.atoms:
            if np.isclose(atom.b, b, atol=tolerance) and np.isclose(atom.y, y, atol=tolerance) and np.isclose(atom.sigma, s, atol=tolerance):
                return atom
        return None
    
    def atomsSimilarTo(self, atom:ZSAtom, threshold=0.95) -> List[ZSAtom]:
        """
        Get a list of atoms similar to the given atom based on its parameters.

        Args:
            atom (ZSAtom): The atom to compare with.
            tolerance (float, optional): The tolerance for the comparison. Defaults to 1e-5.

        Returns:
            List[ZSAtom]: A list of atoms similar to the given atom.
        """
        correlations = self.computeCorrelations(atom())
        print()
        similar_atoms = [self.atoms[i] for i in np.where(correlations > threshold)[0]]
        return similar_atoms
    
    def atomsSimilarToParams(self, b, y, s, tolerance=1e-5) -> List[ZSAtom]:
        """
        Get a list of atoms similar to the given atom based on its parameters.

        Args:
            atom (ZSAtom): The atom to compare with.
            tolerance (float, optional): The tolerance for the comparison. Defaults to 1e-5.

        Returns:
            List[ZSAtom]: A list of atoms similar to the given atom.
        """
        atom = self.getAtomFromParams(b, y, s, tolerance=tolerance)
        if atom is None:
            return []
        return self.atomsSimilarTo(atom, tolerance=tolerance)

    def generateTestSignal(self, signal_length, sparsity_level, snr_level) -> np.ndarray:
        """Generate a test signal as a linear combination of the atoms in the dictionary
        Args:
            signal_length (int): The length of the signal to generate
            sparsity_level (int): The sparsity level of the signal
            snr_level (float): The SNR level between noise and the signal
        Returns:
            signal (np.ndarray): The generated test signal
            atoms_infos (dict): position : ZSAtom corresponding
        """
        assert sparsity_level <= len(self.atoms), "The sparsity level must be less than or equal to the number of atoms in the dictionary"
        assert signal_length >= self.atoms_length, "The signal length must be greater than or equal to the maximum atom length"
        # Generate the coefficients of the linear combination
        rand_atoms_idx = np.random.choice(len(self.atoms), sparsity_level)
        rand_atoms_offsets = np.random.randint(0, signal_length-self.atoms_length+1, sparsity_level)
        # Initialize the test signal
        signal = np.zeros(signal_length)
        atoms_infos = []
        noisesVarToSNR = []
        # Generate the linear combination of the atoms
        for idx, offset in zip(rand_atoms_idx, rand_atoms_offsets) :
            atom_signal = self.atoms[idx].getSignal()
            atom_offset_signal = np.zeros(signal_length)
            atom_offset_signal[offset:offset+len(atom_signal)] = atom_signal
            signal += atom_offset_signal
            b, y, s = self.atoms[idx].params['b'], self.atoms[idx].params['y'], self.atoms[idx].params['sigma']
            atoms_infos.append({'x': offset, 'b': b, 'y': y, 's': s})
            noisesVarToSNR.append(self.atoms[idx].getNoiseVarToSNR(snr_level))
        # Add noise to the signal with a given policy
        # * noiseVar = max(noisesVarToSNR)
        # * noiseVar = min(noisesVarToSNR)
        # * noiseVar = avg(noisesVarToSNR)
        noiseVar = max(noisesVarToSNR)
        noise = np.random.normal(0, np.sqrt(noiseVar), signal_length)
        signal += noise
        signal /= np.linalg.norm(signal)
        return signal, atoms_infos
    
    def generateSignalsDB(self, batch_size:int, signal_length:int, sparsity_levels:List[int], snr_levels:List[float], output_filename:str) -> None:
        """Generate a database of signals with different SNR levels and store it in a JSON file
        Args:
            batch_size (int): The number of signals to generate for each SNR level
            signals_length (int): The length of the signals to generate
            sparsity_level (List[int]): The sparsity levels of the signals
            snr_levels (List[float]): The list of SNR levels to generate the signals
            output_filename (str): The name of the output file to store the signals
        Returns:
            None : it saves the signals in a file
        """
        # Initialize the database dictionary
        db = {}
        db['date'] = get_today_date_str()
        db['batchSize'] = batch_size
        db['snrLevels'] = snr_levels
        db['signalLength'] = signal_length
        db['sparsityLevels'] = sparsity_levels
        db['dictionary'] = str(self)
        # Initialize the signals list
        idx = 0
        signals = []
        for snr, sparsity in product(snr_levels, sparsity_levels) :
            for _ in range(batch_size):
                signal, infos = self.generateTestSignal(signal_length, sparsity, snr)
                result = {
                    'id' : idx,
                    'snr': snr,
                    'sparsity': sparsity,
                    'signal': signal.tolist(),
                    'atoms': infos
                }
                signals.append(result)
                idx += 1
        # Save the signals in a JSON file
        db['signals'] = signals
        json.dump(db, open(output_filename, 'w'), indent=4, default=handle_non_serializable)
        print(f"Signals database saved in {output_filename}")
    
    def atomsDictPositionMatchingErrors(self, atoms_info:List[dict], recov_atoms_info:List[dict]) -> List[int]:
        """Match the original atoms parameters with their recovered ones
        Args:
            atoms_info (List[dict]): The list of the original atoms informations dict
            recov_atoms_info (List[dict]): The list of the recovered atoms informations dict
        Returns:
            List[int]: The list of the matched atoms position errors
        """
        # Compute the matching between the original and the recovered atoms
        position_errors_combinations = [[np.abs(atoms_info[i]['x'] - recov_atoms_info[j]['x']) for j in range(len(recov_atoms_info))] for i in range(len(atoms_info))]
        position_errors = [recov_atoms_info[np.argmin(position_errors_combinations[i])]['x'] - atoms_info[i]['x'] for i in range(len(atoms_info))]
        return position_errors
    
    def computeConvolutions(self, activations) :
        """
        This funcion is the matvec function of the MaskedConvOperator
        that computes the convolution of the activations with the dictionary
        It reconstructs the signal from the activations and the dictionary
        Args:
            activations (np.ndarray): The activations of the signal
        Returns:
            np.ndarray: The reconstructed signal from the activations and the dictionary
        """
        atoms_signals = np.array([atom() for atom in self.atoms])
        activations = np.reshape(activations, (-1, len(self.atoms)))
        convolutions = oaconvolve(activations, atoms_signals.T, mode='full', axes=0).sum(axis=1)
        return convolutions

    def computeCorrelations(self, signal):
        """
        This function computes the correlation of the signal with the dictionary
        using matrix operations to optimize the computation.
        Args:
            signal (np.ndarray): The input signal to correlate with the dictionary
        Returns:
            np.ndarray: The matrix of correlation values between the signal and the dictionary
        """
        atom_signals = np.array([atom()[::-1] for atom in self.atoms])  
        stacked_signals = np.stack([signal] * len(self.atoms), axis=1)  
        all_correlations = oaconvolve(stacked_signals, atom_signals.T, mode='valid', axes=0)
        return all_correlations
    
    def getMaskedConvOperator(self, activation_mask) :
        """Return a LinearOperator that masks and does the convolution."""
        nb_activations = activation_mask.shape[0]
        nb_valid_offset = nb_activations // len(self.atoms)
        nb_samples = nb_valid_offset + self.atoms_length - 1

        # Mask operator : nb_activations x nb_activations
        mask_func = lambda x: x * activation_mask
        mask_operator = LinearOperator(shape=(nb_activations, nb_activations), matvec=mask_func, rmatvec=mask_func)
        
        # Convolution operator : nb_samples x nb_activations
        conv_matvec = partial(self.computeConvolutions)
        conv_rmatvec = partial(self.computeCorrelations)
        conv_operator = LinearOperator(shape=(nb_samples, nb_activations), matvec=conv_matvec, rmatvec=conv_rmatvec)

        # Return the composition of the two operators
        return conv_operator @ mask_operator
    
    #               ______     __    __     ______  
    #              /\  __ \   /\ "-./  \   /\  == \ 
    #              \ \ \/\ \  \ \ \-./\ \  \ \  _-/ 
    #               \ \_____\  \ \_\ \ \_\  \ \_\   
    #                \/_____/   \/_/  \/_/   \/_/                                     

    @time_decorator
    def omp(self, signal:np.ndarray, sparsity_level:int, verbose:bool=False) -> Tuple[np.ndarray, List[dict]]:
        """Orthogonal Matching Pursuit algorithm to recover the sparse signal
        Args:
            signal (np.ndarray): The input signal to recover
            sparsity_level (int): The sparsity level of the signal
        Returns:
            np.ndarray: The matrix of the chosen atoms for the signal
            list: The list of dictionaries containing the parameters of each atom
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        (signal_length,) = signal.shape
        nb_valid_offset = signal_length - self.atoms_length + 1

        # Activation mask parameters 
        nb_activations = nb_valid_offset * len(self.atoms)
        activation_mask = np.zeros((nb_activations,), dtype=np.float64)
        iter = 0
        residual = signal
        while iter < sparsity_level :
            if verbose:
                print("OMP {}/{}".format(iter+1, sparsity_level))

            # Compute the correlation between the signal and the dictionary
            all_correlations = self.computeCorrelations(residual)
            max_corr_idx = all_correlations.argmax()
            activation_mask[max_corr_idx] = 1.0

            # Compute the masked convolution operator
            # This mask takes into account the previous activations
            masked_conv_op = self.getMaskedConvOperator(activation_mask)

            # Solve the LSQR system :
            # Least Squares with QR decomposition
            # A @ x = b with A = masked_conv_op, x = activations, b = signal
            activations, *_ = lsqr(masked_conv_op, signal)
            approx = masked_conv_op @ activations
            residual = signal - approx
            iter += 1

        # Extract the sparse signal from the activations and sort them
        (nnz_indexes,) = np.where(activations)
        nnz_values = activations[nnz_indexes]
        # Sort in descending order
        order = np.argsort(nnz_values)[::-1] 
        nnz_indexes = nnz_indexes[order]
    
        # Extract the atoms and their parameters
        positions_idx, atoms_idx = np.unravel_index(
            nnz_indexes,
            shape=activations.reshape(-1, len(self.atoms)).shape,
        )
        activations = activations.reshape(-1, len(self.atoms))

        infos = list()
        for pos_idx, atom_idx in zip(positions_idx, atoms_idx):
            b, y, s = self.atoms[atom_idx].params['b'], self.atoms[atom_idx].params['y'], self.atoms[atom_idx].params['sigma']
            infos.append({'x': pos_idx, 'b': b, 'y': y, 's': s})

        return approx, infos
    
    def ompFromDict(self, signal_dict:dict, verbose:bool=False) -> dict :
        """Recover the sparse signal from a dictionary of the signal
        Args:
            signal_dict (dict): The dictionary of the signal
        Returns:
            result : OMP results with the dict format
        """
        signal = signal_dict['signal']
        sparsity_level = len(signal_dict['atoms'])
        approx, infos = self.omp(signal, sparsity_level, verbose=verbose)
        mse = np.mean((signal - approx)**2)
        omp_result = {
            'id' : signal_dict['id'],
            'mse' : mse,
            'sparsity' : len(infos),
            'approx' : approx.tolist(),
            'atoms' : infos
        }
        return omp_result
    
    def ompTestBatch(self, batch_size, signals_length, sparsity_level, snr_level, verbose=False) :
        """Test the OMP algorithm on a batch of signals
        Args:
            batch_size (int): The number of signals to generate and test
            signals_length (int): The length of the signals to generate
            sparsity_level (int): The sparsity level of the signals
            snr_level (float): The level of noise to add to the signals
        Returns:
            approximations (np.ndarray): The approximations of the signals
            activations_list (list): The list of the activations of the signals
            infos_list (list): The list of atoms informations dict : the original and the recovered ones
        """
        signals = np.zeros((batch_size, signals_length))
        reconstructions= np.zeros((batch_size, signals_length))
        infos = []
        if verbose :
            print("~~~OMP Test Batch of {} signals ~~~".format(batch_size))
        for i in tqdm(range(batch_size), desc='OMP Test Batch for snr_level = {:.2f}'.format(snr_level)) :
            if verbose :
                print("========= Signal {}/{} =========".format(i+1, batch_size))
            signal, atoms_info = self.generateTestSignal(signals_length, sparsity_level, snr_level)
            approx, recov_atoms_infos = self.omp(signal, sparsity_level, verbose=verbose)
            # Store the signals and the reconstructions
            signals[i] = signal
            reconstructions[i] = approx
            # Compute the Mean Squared Error
            mse = np.mean((signal - approx)**2)
            # Store the informations in a dictionary
            infos.append({'original':atoms_info, 'recovered':recov_atoms_infos, 'mse': mse})

        return signals, reconstructions, infos

    def ompPipelineFromDB(self, input_filename:str, output_filename:str, nb_cores:int, verbose=False) :
        """Create a pipeline of the OMP algorithm from the database of signals.
        Args:
            input_filename (str): The name of the input file containing the signals database
            output_filename (str): The name of the output file to store the results
        Returns:
            None : it saves the results in a file
        """
        with open(input_filename, 'r') as json_file:
            data = json.load(json_file)
            if data is None:
                raise ValueError("The input file is empty or does not contain any data.")
        
        if verbose :
            print(f"OMP Pipeline from {input_filename} with {len(data['signals'])} signals")

        # Extract the signals from the DB
        signals = data['signals']
        # Create the results dictionary
        results = dict()
        results['source'] = input_filename
        results['date'] = get_today_date_str()
        results['algorithm'] = 'Convolutional OMP'
        results['batchSize'] = data['batchSize']
        results['snrLevels'] = data['snrLevels']
        results['signalLength'] = data['signalLength']
        results['sparsityLevels'] = data['sparsityLevels']
        results['dictionary'] = str(self)
        # Parallelize the OMP algorithm on the signals from the DB
        omp_results = Parallel(n_jobs=nb_cores)(delayed(self.ompFromDict)(signal_dict, verbose=verbose) for signal_dict in tqdm(signals, desc='OMP Pipeline from DB'))
        results['omp'] = omp_results
        # Save the results in a JSON file
        json.dump(results, open(output_filename, 'w'), indent=4, default=handle_non_serializable)
        if verbose :
            print(f"OMP Pipeline results saved in {output_filename}")

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

    def mmpdf(self, signal:np.ndarray, sparsity_level:int, connections_level:int, nb_branches:int, verbose:bool=False) -> Tuple[np.ndarray, List[dict]]:
        """ Multipath Matching Pursuit algorithm to recover the sparse signal
        Args:
            signal (np.ndarray): The input signal to recover
            sparsity_level (int): The sparsity level of the signal
        Returns:
            np.ndarray: The matrix of the chosen atoms for the signal
            list: The list of dictionaries containing the parameters of each atom
        """
        mmp_tree = MMPTree(dictionary=self, signal=signal, sparsity=sparsity_level, connections=connections_level)
        mmp_tree.runMMPDF(branches_number=nb_branches, verbose=verbose)
        approx, infos = mmp_tree.getResult()
        return approx, infos

    def mmpdfFromDict(self, signal_dict:dict, connections_level:int, nb_branches:int, verbose:bool=False) -> dict :
        """Recover the sparse signal from a dictionary of the signal
        Args:
            signal_dict (dict): The dictionary of the signal
        Returns:
            result : MMP results with the dict format
        """
        # Extract the signal and the sparsity level
        signal = signal_dict['signal']
        sparsity_level = len(signal_dict['atoms'])
        # Run the MMPDF algorithm
        mmp_tree = MMPTree(dictionary=self, signal=signal, sparsity=sparsity_level, connections=connections_level)
        mmp_tree.runMMPDF(branches_number=nb_branches, verbose=verbose)
        mmp_tree_dict = mmp_tree.buildMMPTreeDict()
        omp_result = {
            'id' : signal_dict['id'],
            'sparsity' : sparsity_level,
            'mmp-tree' : mmp_tree_dict
        }
        return omp_result
    
    def mmpdfPipelineFromDB(self, input_filename:str, output_filename:str, nb_cores:int, verbose=False) :
        """Create a pipeline of the OMP algorithm from the database of signals.
        Args:
            input_filename (str): The name of the input file containing the signals database
            output_filename (str): The name of the output file to store the results
        Returns:
            None : it saves the results in a file
        """
        with open(input_filename, 'r') as json_file:
            data = json.load(json_file)
            if data is None:
                raise ValueError("The input file is empty or does not contain any data.")
        
        if verbose :
            print(f"MMP-DF Pipeline from {input_filename} with {len(data['signals'])} signals")

        # MMP-DF parameters
        connections = 3
        nb_branches = 10

        # Extract the signals from the DB
        signals = data['signals']
        # Create the results dictionary
        results = dict()
        results['source'] = input_filename
        results['date'] = get_today_date_str()
        results['algorithm'] = 'Convolutional MMP-DF'
        results['nbBranches'] = nb_branches
        results['connections'] = connections
        results['batchSize'] = data['batchSize']
        results['snrLevels'] = data['snrLevels']
        results['signalLength'] = data['signalLength']
        results['sparsityLevels'] = data['sparsityLevels']
        results['dictionary'] = str(self)

        # Parallelize the OMP algorithm on the signals from the DB
        mmpdf_results = Parallel(n_jobs=nb_cores)(delayed(self.mmpdfFromDict)(signal_dict, connections_level=connections, nb_branches=nb_branches, verbose=verbose) for signal_dict in tqdm(signals, desc='MMP-DF Pipeline from DB'))
        results['mmp'] = mmpdf_results
        # Save the results in a JSON file
        json.dump(results, open(output_filename, 'w'), indent=4, default=handle_non_serializable)
        if verbose :
            print(f"MMP-DF Pipeline results saved in {output_filename}")