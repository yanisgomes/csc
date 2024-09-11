import datetime
import numpy as np
from typing import List, Dict, Tuple
from csc.atoms import CSCAtom
from scipy import optimize

def handle_non_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError("Unserializable object {} of type {}".format(obj, type(obj)))

def get_today_date_str():
    # Récupérer la date du jour
    today = datetime.datetime.now()
    # Formater la date en 'AAMMDD'
    return today.strftime('%y%m%d')

def computeMaxTruePositives(dictionary, signal_length, true_atoms:List[Dict], approx_atoms:List[Dict], pos_err_threshold:float, corr_err_threshold:float, verbose:bool=False) -> int:
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
        CSC_atom = CSCAtom.from_dict(atom)
        CSC_atom.padBothSides(dictionary.getAtomsLength())
        true_signals.append(CSC_atom.getAtomInSignal(signal_length, atom['x']))
    approx_signals = []
    for atom in approx_atoms:
        CSC_atom = CSCAtom.from_dict(atom)
        CSC_atom.padBothSides(dictionary.getAtomsLength())
        approx_signals.append(CSC_atom.getAtomInSignal(signal_length, atom['x']))
    correlation_matrix = np.zeros((len(true_atoms), len(approx_atoms)))

    for i, true_signal in enumerate(true_signals):
        for j, approx_signal in enumerate(approx_signals):
            correlation_matrix[i, j] = np.abs(np.correlate(true_signal, approx_signal, mode='valid')[0])

    # Define the cost matrix based on position and correlation thresholds
    cost_matrix = np.ones_like(pos_cost_matrix)
    cost_matrix[(pos_cost_matrix <= pos_err_threshold) & (correlation_matrix >= corr_err_threshold)] = 0

    # Compute the optimal matching
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
    
    # Get the matched atoms tuples
    matched_atoms = [(true_atoms[i], approx_atoms[j]) for i, j in zip(row_ind, col_ind)]

    # Compute the maximum number of true positives
    max_tp = np.sum(cost_matrix[row_ind, col_ind] == 0)

    return max_tp, matched_atoms

def getMatchedAtoms(dictionary, signal_length, true_atoms:List[Dict], approx_atoms:List[Dict], pos_err_threshold:float, corr_err_threshold:float, verbose:bool=False) -> List[Tuple[Dict, Dict]]:
    """
    Get the matched atoms between the true and approximation dictionaries.
    Args:
        true_atoms (List[Dict]): Atoms of the true dictionary.
        approx_atoms (List[Dict]): Atoms of the approximation dictionary.
        pos_err_threshold (float): Position error threshold for matching.
        corr_err_threshold (float): Correlation threshold for matching.
    Returns:
        List[Tuple[Dict, Dict]]: List of matched atoms tuples.
    """
    # Extract positions and compute cost matrix for positions
    true_positions = np.array([atom['x'] for atom in true_atoms])
    approx_positions = np.array([atom['x'] for atom in approx_atoms])
    pos_cost_matrix = np.abs(true_positions[:, np.newaxis] - approx_positions)

    # Compute correlation matrix
    true_signals = []
    for atom in true_atoms:
        CSC_atom = CSCAtom.from_dict(atom)
        CSC_atom.padBothSides(dictionary.getAtomsLength())
        true_signals.append(CSC_atom.getAtomInSignal(signal_length, atom['x']))
    approx_signals = []
    for atom in approx_atoms:
        CSC_atom = CSCAtom.from_dict(atom)
        CSC_atom.padBothSides(dictionary.getAtomsLength())
        approx_signals.append(CSC_atom.getAtomInSignal(signal_length, atom['x']))
    correlation_matrix = np.zeros((len(true_atoms), len(approx_atoms)))

    for i, true_signal in enumerate(true_signals):
        for j, approx_signal in enumerate(approx_signals):
            correlation_matrix[i, j] = np.abs(np.correlate(true_signal, approx_signal, mode='valid')[0])

    # Define the cost matrix based on position and correlation thresholds
    cost_matrix = np.ones_like(pos_cost_matrix)
    cost_matrix[(pos_cost_matrix <= pos_err_threshold) & (correlation_matrix >= corr_err_threshold)] = 0

    # Compute the optimal matching
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)

    # Get the matched atoms tuples
    matched_atoms = [(true_atoms[i], approx_atoms[j]) for i, j in zip(row_ind, col_ind)]

    return matched_atoms