import os
import math
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

from alphacsc.update_z import update_z

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

    def getLocalDictionary(self) -> np.ndarray:
        return np.array([atom() for atom in self.atoms])
        
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
    
    def correlationFromDicts(self, atom1_dict:dict, atom2_dict:dict, signal_length:int) -> float:
        """Compute the correlation between two atoms from their parameters
        Args:
            atom1_dict (dict): The parameters of the first atom
            atom2_dict (dict): The parameters of the second atom
        Returns:
            float: The correlation between the two atoms
        """
        # Build atom1 signal
        atom1 = ZSAtom.from_dict(atom1_dict)
        atom1.padBothSides(self.atoms_length)
        atom1_signal = atom1.getAtomInSignal(signal_length, offset=atom1_dict['x'])
        
        # Build atom2 signal
        atom2 = ZSAtom.from_dict(atom2_dict)
        atom2.padBothSides(self.atoms_length)
        atom2_signal = atom2.getAtomInSignal(signal_length, offset=atom2_dict['x'])

        (correlation,) = np.correlate(atom1_signal, atom2_signal, mode='valid')
        return correlation

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
        
        non_zero_index = np.argwhere(np.abs(signal) > ZSAtom.SUPPORT_THRESHOLD)
        non_zero_signal = signal[non_zero_index]
        signal_power = np.var(non_zero_signal)   
        noise_var = signal_power / (10 ** (snr_level / 10))
        noise = np.random.normal(0, np.sqrt(noise_var), signal_length)
        noisy_signal = signal + noise
        return noisy_signal, atoms_infos
    
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

    
    #      ______                 __             _                __
    #     / ____/___  ____  _____/ /__________ _(_)___  ___  ____/ /
    #    / /   / __ \/ __ \/ ___/ __/ ___/ __ `/ / __ \/ _ \/ __  / 
    #   / /___/ /_/ / / / (__  ) /_/ /  / /_/ / / / / /  __/ /_/ /  
    #   \____/\____/_/ /_/____/\__/_/   \__,_/_/_/ /_/\___/\__,_/   
    #                                                               

    def generateConstrainedTestSignal(self, signal_length:int, sparsity_level:int, snr_level:int, pos_err_threshold:int, corr_err_threshold:float) -> np.ndarray:
        """Generate a test signal as a linear combination of the atoms in the dictionary
        Args:
            signal_length (int): The length of the signal to generate
            sparsity_level (int): The sparsity level of the signal
            snr_level (int): The SNR level between noise and the signal in decibels
            pos_err_threshold (int): The maximum position error between two similar atoms
            corr_err_threshold (float): The maximum correlation error between two similar atoms
        Returns:
            signal (np.ndarray): The generated test signal
            atoms_infos (dict): position : ZSAtom corresponding
        """
        assert sparsity_level <= len(self.atoms), "The sparsity level must be less than or equal to the number of atoms in the dictionary"
        assert signal_length >= self.atoms_length, "The signal length must be greater than or equal to the maximum atom length"
        
        # Initialize the test signal
        signal = np.zeros(signal_length)
        atom_signals = []
        atom_positions = []
        atoms_infos = []

        while len(atoms_infos) < sparsity_level:
            atom_idx = np.random.choice(len(self.atoms))
            atom_position = np.random.randint(0, signal_length-self.atoms_length+1)

            atom_signal = self.atoms[atom_idx].getSignal()
            atom_full_signal = np.zeros(signal_length)
            atom_full_signal[atom_position:atom_position+len(atom_signal)] = atom_signal

            # Check if similar atoms are not already in the signal
            similar_atom_found = False
            for other_atom_position, other_atom_signal in zip(atom_positions, atom_signals):
                corr = np.correlate(atom_full_signal, other_atom_signal, mode='valid')
                corr = corr / (np.linalg.norm(atom_full_signal) * np.linalg.norm(other_atom_signal))  # Normalized correlation
                abs_pos_diff = np.abs(atom_position - other_atom_position)
                if abs_pos_diff < pos_err_threshold or np.max(corr) > corr_err_threshold:
                    similar_atom_found = True
                    break
        
            if similar_atom_found:
                continue
            
            signal += atom_full_signal
            atom_positions.append(atom_position)
            atom_signals.append(atom_full_signal)
            atoms_infos.append({'x': atom_position, 'b': self.atoms[atom_idx].params['b'], 'y': self.atoms[atom_idx].params['y'], 's': self.atoms[atom_idx].params['sigma']})
        
        non_zero_index = np.argwhere(np.abs(signal) > ZSAtom.SUPPORT_THRESHOLD)
        non_zero_signal = signal[non_zero_index]
        signal_power = np.var(non_zero_signal)   
        noise_var = signal_power / (10 ** (snr_level / 10))
        noise = np.random.normal(0, np.sqrt(noise_var), signal_length)
        noisy_signal = signal + noise
        return noisy_signal, atoms_infos

    def generateConstrainedSignalsDB(self, batch_size:int, signal_length:int, sparsity_levels:List[int], snr_levels:List[float], pos_err_threshold:float, corr_err_threshold:float, output_filename:str) -> None:
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
        db['posErrThreshold'] = pos_err_threshold
        db['corrErrThreshold'] = corr_err_threshold
        db['dictionary'] = str(self)
        # Initialize the signals list
        idx = 0
        signals = []
        for snr, sparsity in product(snr_levels, sparsity_levels) :
            for _ in range(batch_size):
                signal, infos = self.generateConstrainedTestSignal(signal_length, sparsity, snr, pos_err_threshold, corr_err_threshold)
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
    
    def getActivationIdxFromAtom(self, atom:ZSAtom, offset:int) -> int:
        """Return the index of the activation corresponding to the atom and the offset
        Args:
            atom (ZSAtom): The atom to find in the dictionary
            offset (int): The offset of the atom in the signal
        Returns:
            int: The index of the activation corresponding to the atom and the offset
        """
        atom_idx = self.atoms.index(atom)
        return offset*len(self.atoms) + atom_idx
    
    def getActivationIdxFromParams(self, params_dict:dict) -> int:
        """Return the index of the activation corresponding to the atom parameters
        Args:
            params_dict (dict): The parameters of the atom to find in the dictionary
        Returns:
            int: The index of the activation corresponding to the atom parameters
        """
        atom = ZSAtom.from_dict(params_dict)
        offset = params_dict['x']
        return self.getActivationIdxFromAtom(atom, offset)

    def getSignalProjectionFromAtoms(self, signal:np.ndarray, atoms_dicts:List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Project the activations on the dictionary
        Args:
            signal (np.ndarray): The input signal to project on the dictionary
            atoms_dicts (List[dict]): The list of the atoms parameters to project
        Returns:
            approx (np.ndarray): The projection of the activations on the dictionary
            activations (np.ndarray): The activations of the signal
        """
        # Activation mask parameters 
        signal_length = len(signal)
        nb_valid_offset = signal_length - self.atoms_length + 1
        nb_activations = nb_valid_offset * len(self.atoms)
        activation_mask = np.zeros((nb_activations,), dtype=np.float64)
        for atom_dict in atoms_dicts :
            activation_idx = self.getActivationIdxFromParams(atom_dict)
            activation_mask[activation_idx] = 1.0
        # Compute the masked convolution operator
        masked_conv_op = self.getMaskedConvOperator(activation_mask)
        # Solve the LSQR system :
        # Least Squares with QR decomposition
        # A @ x = b with A = masked_conv_op, x = activations, b = signal
        activations, *_ = lsqr(masked_conv_op, signal)
        approx = masked_conv_op @ activations
        return approx, activations
    
    def getAtomFromActivationIdx(self, activation_idx:int) -> Tuple[ZSAtom, int]:
        """Return the atom and the offset corresponding to the activation index
        Args:
            activation_idx (int): The index of the activation in the dictionary
        Returns:
            Tuple[ZSAtom, int]: The atom and the offset corresponding to the activation index
        """
        atom_idx = activation_idx % len(self.atoms)
        offset = activation_idx // len(self.atoms)
        return self.atoms[atom_idx], offset

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
    
    def ompSparVarFromDict(self, signal_dict:dict, max_sparsity:int, verbose:bool=False) -> dict:
        """Orthogonal Matching Pursuit algorithm to recover the sparse signal for each sparsity_level <= max_sparsity_level
        Args:
            signal (np.ndarray): The input signal to recover
            sparsity_level (int): The sparsity level of the signal
        Returns:
            omp_sparVar_results : OMP results with the dict format
        """
        signal = signal_dict['signal']

        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        (signal_length,) = signal.shape
        nb_valid_offset = signal_length - self.atoms_length + 1

        # Activation mask parameters 
        nb_activations = nb_valid_offset * len(self.atoms)
        activation_mask = np.zeros((nb_activations,), dtype=np.float64)
        iter = 0
        residual = signal
        omp_sparVar_list = []
        t0 = time.time()
        while iter < max_sparsity :
            if verbose:
                print("OMP {}/{}".format(iter+1, max_sparsity))

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

            atoms = list()
            for pos_idx, atom_idx in zip(positions_idx, atoms_idx):
                b, y, s = self.atoms[atom_idx].params['b'], self.atoms[atom_idx].params['y'], self.atoms[atom_idx].params['sigma']
                atoms.append({'x': pos_idx, 'b': b, 'y': y, 's': s})

            omp_sparVar_list.append(
                {
                    'mse' : np.mean((signal - approx)**2),
                    'atoms' : atoms,
                    'delay' : time.time() - t0
                }
            )
        
        omp_sparVar_results = {
            'id' : signal_dict['id'],
            'snr' : signal_dict['snr'],
            'results' : omp_sparVar_list
        }
                
        return omp_sparVar_results
    
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

    def ompSparVarPipelineFromDB(self, input_filename:str, output_filename:str, nb_cores:int, max_sparsity:int, verbose=False) :
        """Create a pipeline of the OMP algorithm from the database of signals.
        Args:
            input_filename (str): The name of the input file containing the signals database
            output_filename (str): The name of the output file to store the results
            nb_cores (int): The number of cores to use for the parallelization
            max_sparsity (int): The maximum sparsity level to test
        Returns:
            None : it saves the results in a file
        """
        with open(input_filename, 'r') as json_file:
            data = json.load(json_file)
            if data is None:
                raise ValueError("The input file is empty or does not contain any data.")
        
        if verbose :
            print(f"OMP SparVar Pipeline from {input_filename} with {len(data['signals'])} signals")

        # Extract the signals from the DB
        signals = data['signals']
        # Create the results dictionary
        results = dict()
        results['source'] = input_filename
        results['date'] = get_today_date_str()
        results['algorithm'] = 'Convolutional OMP'
        results['batchSize'] = data['batchSize']
        results['snrLevels'] = data['snrLevels']
        results['maxSparsityLevel'] = max_sparsity
        results['signalLength'] = data['signalLength']
        results['sparsityLevels'] = data['sparsityLevels']
        results['dictionary'] = str(self)
        # Parallelize the OMP algorithm on the signals from the DB
        omp_results = Parallel(n_jobs=nb_cores)(delayed(self.ompSparVarFromDict)(signal_dict, max_sparsity=max_sparsity, verbose=verbose) for signal_dict in tqdm(signals, desc='OMP SparVar Pipeline from DB'))
        results['omp'] = omp_results
        # Save the results in a JSON file
        json.dump(results, open(output_filename, 'w'), indent=4, default=handle_non_serializable)
        if verbose :
            print(f"OMP Pipeline results saved in {output_filename}")

    #                __    __     ______  
    #               /\ "-./  \   /\  == \ 
    #               \ \ \-./\ \  \ \  _-/ 
    #                \ \_\ \ \_\  \ \_\   
    #                 \/_/  \/_/   \/_/                                     

    @time_decorator
    def mp(self, signal: np.ndarray, sparsity_level: int, verbose: bool = False) -> Tuple[np.ndarray, List[dict]]:
        """Matching Pursuit algorithm to recover the sparse signal
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
        nb_atoms = len(self.atoms)

        # Activation mask parameters
        nb_activations = nb_valid_offset * len(self.atoms)
        activation_mask = np.zeros((nb_activations,), dtype=np.float64)
        iter = 0
        residual = signal
        selected_atoms = []

        while iter < sparsity_level:
            if verbose:
                print("MP {}/{}".format(iter + 1, sparsity_level))

            # Compute the correlation between the residual and the dictionary
            all_correlations = self.computeCorrelations(residual)
            max_corr_idx_flat = all_correlations.argmax()
            max_corr_idx = np.unravel_index(max_corr_idx_flat, all_correlations.shape)
            max_corr_value = all_correlations[max_corr_idx]
            activation_mask[max_corr_idx_flat] += max_corr_value

            # Compute the masked convolution operator
            # This mask takes into account the previous activations
            masked_conv_op = self.getMaskedConvOperator(activation_mask)

            # Update the residual
            approx = masked_conv_op @ activation_mask
            residual = signal - approx
            iter += 1

        # Extract the sparse signal from the activations and sort them
        (nnz_indexes,) = np.where(activation_mask)
        nnz_values = activation_mask[nnz_indexes]
        # Sort in descending order
        order = np.argsort(nnz_values)[::-1] 
        nnz_indexes = nnz_indexes[order]
    
        # Extract the atoms and their parameters
        positions_idx, atoms_idx = np.unravel_index(
            nnz_indexes,
            shape=activation_mask.reshape(-1, len(self.atoms)).shape,
        )

        infos = list()
        for pos_idx, atom_idx in zip(positions_idx, atoms_idx):
            b, y, s = self.atoms[atom_idx].params['b'], self.atoms[atom_idx].params['y'], self.atoms[atom_idx].params['sigma']
            infos.append({'x': pos_idx, 'b': b, 'y': y, 's': s})

        return approx, infos

    def mpFromDict(self, signal_dict: dict, verbose: bool = False) -> dict:
        """Recover the sparse signal from a dictionary of the signal
        Args:
            signal_dict (dict): The dictionary of the signal
        Returns:
            result : MP results with the dict format
        """
        signal = signal_dict['signal']
        sparsity_level = len(signal_dict['atoms'])
        approx, infos = self.mp(signal, sparsity_level, verbose=verbose)
        mse = np.mean((signal - approx)**2)
        mp_result = {
            'id': signal_dict['id'],
            'mse': mse,
            'sparsity': len(infos),
            'approx': approx.tolist(),
            'atoms': infos
        }
        return mp_result

    def mpSparVarFromDict(self, signal_dict:dict, max_sparsity:int, verbose:bool=False) -> dict:
        """Matching Pursuit algorithm to recover the sparse signal for each sparsity_level <= max_sparsity_level
        Args:
            signal (np.ndarray): The input signal to recover
            sparsity_level (int): The sparsity level of the signal
        Returns:
            mp_sparVar_results : The results of the MP algorithm with the dict format
        """
        signal = signal_dict['signal']

        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        (signal_length,) = signal.shape
        nb_valid_offset = signal_length - self.atoms_length + 1

        # Activation mask parameters 
        nb_activations = nb_valid_offset * len(self.atoms)
        activation_mask = np.zeros((nb_activations,), dtype=np.float64)
        iter = 0
        residual = signal
        mp_sparVar_list = []
        t0 = time.time()
        while iter < max_sparsity :
            if verbose:
                print("MP {}/{}".format(iter + 1, max_sparsity))

            # Compute the correlation between the residual and the dictionary
            all_correlations = self.computeCorrelations(residual)
            max_corr_idx_flat = all_correlations.argmax()
            max_corr_idx = np.unravel_index(max_corr_idx_flat, all_correlations.shape)
            max_corr_value = all_correlations[max_corr_idx]
            activation_mask[max_corr_idx_flat] += max_corr_value

            # Compute the masked convolution operator
            # This mask takes into account the previous activations
            masked_conv_op = self.getMaskedConvOperator(activation_mask)

            # Update the residual
            approx = masked_conv_op @ activation_mask
            residual = signal - approx
            iter += 1

            # Extract the sparse signal from the activations and sort them
            (nnz_indexes,) = np.where(activation_mask)
            nnz_values = activation_mask[nnz_indexes]
            # Sort in descending order
            order = np.argsort(nnz_values)[::-1] 
            nnz_indexes = nnz_indexes[order]
        
            # Extract the atoms and their parameters
            positions_idx, atoms_idx = np.unravel_index(
                nnz_indexes,
                shape=activation_mask.reshape(-1, len(self.atoms)).shape,
            )

            atoms = list()
            for pos_idx, atom_idx in zip(positions_idx, atoms_idx):
                b, y, s = self.atoms[atom_idx].params['b'], self.atoms[atom_idx].params['y'], self.atoms[atom_idx].params['sigma']
                atoms.append({'x': pos_idx, 'b': b, 'y': y, 's': s})

            mp_sparVar_list.append(
                {
                    'mse' : np.mean((signal - approx)**2),
                    'atoms' : atoms,
                    'delay' : time.time() - t0
                }
            )
        
        mp_sparVar_results = {
            'id' : signal_dict['id'],
            'snr' : signal_dict['snr'],
            'results' : mp_sparVar_list
        }
                
        return mp_sparVar_results
    
    def mpPipelineFromDB(self, input_filename: str, output_filename: str, nb_cores: int, verbose=False):
        """Create a pipeline of the MP algorithm from the database of signals.
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
        
        if verbose:
            print(f"MP Pipeline from {input_filename} with {len(data['signals'])} signals")

        # Extract the signals from the DB
        signals = data['signals']
        # Create the results dictionary
        results = dict()
        results['source'] = input_filename
        results['date'] = get_today_date_str()
        results['algorithm'] = 'Convolutional MP'
        results['batchSize'] = data['batchSize']
        results['snrLevels'] = data['snrLevels']
        results['signalLength'] = data['signalLength']
        results['sparsityLevels'] = data['sparsityLevels']
        results['dictionary'] = str(self)
        # Parallelize the MP algorithm on the signals from the DB
        mp_results = Parallel(n_jobs=nb_cores)(delayed(self.mpFromDict)(signal_dict, verbose=verbose) for signal_dict in tqdm(signals, desc='MP Pipeline from DB'))
        results['mp'] = mp_results
        # Save the results in a JSON file
        json.dump(results, open(output_filename, 'w'), indent=4, default=handle_non_serializable)
        if verbose:
            print(f"MP Pipeline results saved in {output_filename}")

    def mpSparVarPipelineFromDB(self, input_filename:str, output_filename:str, nb_cores:int, max_sparsity:int, verbose=False) :
        """Create a pipeline of the MP algorithm from the database of signals.
        Args:
            input_filename (str): The name of the input file containing the signals database
            output_filename (str): The name of the output file to store the results
            nb_cores (int): The number of cores to use for the parallelization
            max_sparsity (int): The maximum sparsity level to test
        Returns:
            None : it saves the results in a file
        """
        with open(input_filename, 'r') as json_file:
            data = json.load(json_file)
            if data is None:
                raise ValueError("The input file is empty or does not contain any data.")
        
        if verbose :
            print(f"MP SparVar Pipeline from {input_filename} with {len(data['signals'])} signals")

        # Extract the signals from the DB
        signals = data['signals']
        # Create the results dictionary
        results = dict()
        results['source'] = input_filename
        results['date'] = get_today_date_str()
        results['algorithm'] = 'Convolutional MP'
        results['batchSize'] = data['batchSize']
        results['snrLevels'] = data['snrLevels']
        results['maxSparsityLevel'] = max_sparsity
        results['signalLength'] = data['signalLength']
        results['sparsityLevels'] = data['sparsityLevels']
        results['dictionary'] = str(self)
        # Parallelize the MP algorithm on the signals from the DB
        mp_results = Parallel(n_jobs=nb_cores)(delayed(self.mpSparVarFromDict)(signal_dict, max_sparsity=max_sparsity, verbose=verbose) for signal_dict in tqdm(signals, desc='MP SparVar Pipeline from DB'))
        results['mp'] = mp_results
        # Save the results in a JSON file
        json.dump(results, open(output_filename, 'w'), indent=4, default=handle_non_serializable)
        if verbose :
            print(f"MP Pipeline results saved in {output_filename}")


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

    def mmpdf(self, signal:np.ndarray, sparsity_level:int, connections_level:int, dissimilarity:float, nb_branches:int, verbose:bool=False) -> Tuple[np.ndarray, List[dict]]:
        """ Multipath Matching Pursuit algorithm to recover the sparse signal
        Args:
            signal (np.ndarray): The input signal to recover
            sparsity_level (int): The sparsity level of the signal
        Returns:
            np.ndarray: The matrix of the chosen atoms for the signal
            list: The list of dictionaries containing the parameters of each atom
        """
        mmp_tree = MMPTree(dictionary=self, signal=signal, sparsity=sparsity_level, connections=connections_level, dissimilarity=dissimilarity)
        mmp_tree.runMMPDF(branches_number=nb_branches, verbose=verbose)
        approx, infos = mmp_tree.getResult()
        return approx, infos

    def mmpdfTreeFromDict(self, signal_dict:dict, connections_level:int, dissimilarity:float, nb_branches:int, verbose:bool=False) -> dict :
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
        mmp_tree = MMPTree(dictionary=self, signal=signal, sparsity=sparsity_level, connections=connections_level, dissimilarity=dissimilarity)
        mmp_tree.runMMPDF(branches_number=nb_branches, verbose=verbose)
        mmp_tree_dict = mmp_tree.buildMMPTreeDict()
        mmp_result = {
            'id' : signal_dict['id'],
            'sparsity' : sparsity_level,
            'mmp-tree' : mmp_tree_dict
        }
        return mmp_result
    
    def mmpdfSparVarFromDict(self, signal_dict:dict, connections_level:int, dissimilarity:float, nb_branches:int, max_sparsity:int, verbose:bool=False) -> dict :
        """Recover the sparse signal from a dictionary of the signal 
           returns a list of the result for each candidate_sparsity level <= max_sparsity
        Args:
            signal_dict (dict): The dictionary of the signal
        Returns:
            result : MMP results with the list format
        
        """
        mmpdf_sparVar_list = []
        signal = signal_dict['signal']
        # Iterate over the sparsity levels <= max_sparsity
        for candidate_sparsity in range(1, max_sparsity+1):
            # Run the MMPDF algorithm
            mmp_tree = MMPTree(dictionary=self, signal=signal, sparsity=candidate_sparsity, connections=connections_level, dissimilarity=dissimilarity)
            mmp_tree.runMMPDF(branches_number=nb_branches, verbose=verbose)
            mmp_result = mmp_tree.buildMMPDFResultDict()
            mmpdf_sparVar_list.append(mmp_result)

        mmpdf_sparVar_results = {
            'id' : signal_dict['id'],
            'snr' : signal_dict['snr'],
            'results' : mmpdf_sparVar_list
        }
        return mmpdf_sparVar_results
    
    def mmpdfPipelineFromDB(self, input_filename:str, output_filename:str, nb_cores:int, connections:int=3, dissimilarity:float=0.8, branches:int=10, verbose=False) :
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

        # Extract the signals from the DB
        signals = data['signals']
        # Create the results dictionary
        results = dict()
        results['source'] = input_filename
        results['date'] = get_today_date_str()
        results['algorithm'] = 'Convolutional MMP-DF'
        results['nbBranches'] = branches
        results['connections'] = connections
        results['dissimilarity'] = dissimilarity
        results['batchSize'] = data['batchSize']
        results['snrLevels'] = data['snrLevels']
        results['signalLength'] = data['signalLength']
        results['sparsityLevels'] = data['sparsityLevels']
        results['dictionary'] = str(self)

        # Parallelize the OMP algorithm on the signals from the DB
        mmpdf_results = Parallel(n_jobs=nb_cores)(delayed(self.mmpdfTreeFromDict)(signal_dict, connections_level=connections, dissimilarity=dissimilarity, nb_branches=branches, verbose=verbose) for signal_dict in tqdm(signals, desc='MMP-DF Pipeline from DB'))
        results['mmp'] = mmpdf_results
        # Save the results in a JSON file
        json.dump(results, open(output_filename, 'w'), indent=4, default=handle_non_serializable)
        if verbose :
            print(f"MMP-DF Pipeline results saved in {output_filename}")

    def mmpdfSparVarPipelineFromDB(self, input_filename:str, output_filename:str, nb_cores:int, connections:int=3, dissimilarity:float=0.8, branches:int=10, max_sparsity:int=10, verbose=False) :
        """Create a pipeline of the MMP-DF algorithm from the database of signals.
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

        # Extract the signals from the DB
        signals = data['signals']
        # Create the results dictionary
        results = dict()
        results['source'] = input_filename
        results['date'] = get_today_date_str()
        results['algorithm'] = 'Convolutional MMP-DF'
        results['nbBranches'] = branches
        results['connections'] = connections
        results['dissimilarity'] = dissimilarity
        results['maxSparsityLevel'] = max_sparsity
        results['batchSize'] = data['batchSize']
        results['snrLevels'] = data['snrLevels']
        results['signalLength'] = data['signalLength']
        results['sparsityLevels'] = data['sparsityLevels']
        results['dictionary'] = str(self)

        # Parallelize the OMP algorithm on the signals from the DB
        mmpdf_results = Parallel(n_jobs=nb_cores)(delayed(self.mmpdfSparVarFromDict)(signal_dict, connections_level=connections, dissimilarity=dissimilarity, nb_branches=branches, max_sparsity=max_sparsity, verbose=verbose) for signal_dict in tqdm(signals, desc='MMP-DF SparVar Pipeline from DB'))
        results['mmp'] = mmpdf_results
        # Save the results in a JSON file
        json.dump(results, open(output_filename, 'w'), indent=4, default=handle_non_serializable)
        if verbose :
            print(f"MMP-DF SparVar Pipeline results saved in {output_filename}")


#             888           888                      e88~-_  ,d88~~\  e88~-_  
#     /~~~8e  888 888-~88e  888-~88e   /~~~8e       d888   \ 8888    d888   \ 
#         88b 888 888  888b 888  888       88b ____ 8888     `Y88b   8888     
#    e88~-888 888 888  8888 888  888  e88~-888      8888      `Y88b, 8888     
#   C888  888 888 888  888P 888  888 C888  888      Y888   /    8888 Y888   / 
#    "88_-888 888 888-_88"  888  888  "88_-888       "88_-~  \__88P'  "88_-~  
#                 888                                                         
#   

    def alphaCSCResultFromDict(self, signal_dict: dict, nb_activations: int, verbose: bool = False, tolerance: float = 20.0) -> dict:
        signal = np.array(signal_dict['signal'])
        D = self.getLocalDictionary()

        if signal.ndim == 1:
            signal = signal[np.newaxis, :]

        lmbda = 5e-5  # initial small lambda
        activations = None
        last_nnz_activations = None
        target_activations = nb_activations
        tolerance_margin = math.ceil(tolerance / 100 * target_activations)
        iter = 0

        while True :
            activations = update_z(signal, D, lmbda).squeeze()  # Assuming update_z returns the activations array
            activations = activations.flatten()
            nnz_indexes, = np.nonzero(activations)
            len_activations = len(nnz_indexes)

            if len_activations > 0:
                last_nnz_activations = activations[nnz_indexes]

            if verbose:
                print(f'Iteration {iter}: lambda = {lmbda:.2e}, Number of Activations = {len_activations}')

            # Break condition with tolerance margin
            if len_activations - target_activations >= 0 and len_activations - target_activations <= tolerance_margin:
                break

            # Break condition on the number of iterations
            if iter >= 200 :
                activations = last_nnz_activations
                break

            # Update lambda based on the difference between current and target activations
            if len_activations > target_activations:
                last_lmbda = lmbda
                lmbda *= 1 + 0.08*(len_activations - target_activations) / target_activations
            else:
                last_lmbda_coeff = 0.95
                lmbda = (1 - last_lmbda_coeff) * lmbda + last_lmbda_coeff * last_lmbda
            iter += 1

        # Post-processing of activations
        nnz_values = activations[nnz_indexes]
        order = np.argsort(nnz_values)[::-1]
        nnz_indexes_sorted = nnz_indexes[order]

        # Extract atoms and their parameters
        positions_idx, atoms_idx = np.unravel_index(nnz_indexes_sorted, shape=activations.reshape(-1, len(D)).shape)

        results = list()
        for pos_idx, atom_idx in zip(positions_idx, atoms_idx):
            b, y, s = self.atoms[atom_idx].params['b'], self.atoms[atom_idx].params['y'], self.atoms[atom_idx].params['sigma']
            results.append({'x': pos_idx, 'b': b, 'y': y, 's': s})

        return results
