import time
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

    def getAtoms(self) -> List[ZSAtom] :
        return self.atoms
    
    def getAtomsLength(self) -> int :
        return self.atoms_length

    def generateTestSignal(self, signal_length, sparsity_level, noise_level) -> np.ndarray:
        """Generate a test signal as a linear combination of the atoms in the dictionary
        Args:
            signal_length (int): The length of the signal to generate
            sparsity_level (int): The sparsity level of the signal
            noise_level (float): The level of noise to add to the signal
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
        # Generate the linear combination of the atoms
        for idx, offset in zip(rand_atoms_idx, rand_atoms_offsets) :
            atom_signal = self.atoms[idx].getSignal()
            atom_offset_signal = np.zeros(signal_length)
            atom_offset_signal[offset:offset+len(atom_signal)] = atom_signal
            signal += atom_offset_signal
            b, y, s = self.atoms[idx].params['b'], self.atoms[idx].params['y'], self.atoms[idx].params['sigma']
            atoms_infos.append({'x': offset, 'b': b, 'y': y, 's': s})
        # Add noise to the signal
        noise = noise_level * np.random.randn(signal_length)
        signal += noise
        return signal, atoms_infos
    
    def generateTestSignalBatch(self, signal_length, sparsity_level, noise_level, batch_size) -> np.ndarray:
        """Generate a batch of test signals as linear combinations of the atoms in the dictionary
        Args:
            nb_atoms (int): The number of atoms to use in the linear combination
            noise_level (float): The level of noise to add to the signal
            nb_samples (int): The number of samples to generate
        Returns:
            np.ndarray: The generated batch of test signals
        """
        signals = np.zeros((batch_size, signal_length))
        signals_atoms_infos = []
        for i in range(batch_size):
            signal, atoms_infos = self.generateTestSignal(signal_length, sparsity_level, noise_level)
            signals[i] = signal
            signals_atoms_infos.append(atoms_infos)
        return signals, signals_atoms_infos
    
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
    
    @time_decorator
    def ompTestBatch(self, batch_size, signals_length, sparsity_level, noise_level, verbose=False) :
        """Test the OMP algorithm on a batch of signals
        Args:
            batch_size (int): The number of signals to generate and test
            signals_length (int): The length of the signals to generate
            sparsity_level (int): The sparsity level of the signals
            noise_level (float): The level of noise to add to the signals
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
        for i in tqdm(range(batch_size), desc='OMP Test Batch for noise_level = {:.2f}'.format(noise_level)) :
            if verbose :
                print("========= Signal {}/{} =========".format(i+1, batch_size))
            signal, atoms_info = self.generateTestSignal(signals_length, sparsity_level, noise_level)
            approx, recov_atoms_infos = self.omp(signal, sparsity_level, verbose=verbose)
            # Store the signals and the reconstructions
            signals[i] = signal
            reconstructions[i] = approx
            # Compute the Mean Squared Error
            mse = np.mean((signal - approx)**2)
            # Store the informations in a dictionary
            infos.append({'original':atoms_info, 'recovered':recov_atoms_infos, 'mse': mse})

        return signals, reconstructions, infos

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
    
    def ompPositionErrorBatch(self, sparsity_level:int, noise_level:int, batch_size:int, verbose:bool=False) -> Counter:
        """Compute the histogram of the position errors for the OMP algorithm
        Args:
            sparsity_level (int): The sparsity level of the signals
            noise_level (float): The level of noise to add to the signals
            batch_size (int): The number of signals to generate and test
        Returns:
            pos_err_counter (Counter): The histogram of the position errors
        """
        _, _, infos = self.ompTestBatch(batch_size, self.atoms_length*2, sparsity_level, noise_level, verbose=verbose)
        pos_err_collection = []
        for info in infos:
            pos_err_collection.extend(self.atomsDictPositionMatchingErrors(info['original'], info['recovered']))
        pos_err_counter = Counter(pos_err_collection)
        return pos_err_counter

    def ompPositionErrorPipeline(self, sparsity_level:int, batch_size:int, cores=50, verbose=False):
        noise_levels = np.concatenate((np.arange(0.0, 0.11, 0.01), np.arange(0.12, 0.22, 0.02))) 
        results = Parallel(n_jobs=cores)(delayed(self.ompPositionErrorBatch)(sparsity_level, noise, batch_size, verbose=verbose) for noise in noise_levels)
        return dict(zip(noise_levels, results))