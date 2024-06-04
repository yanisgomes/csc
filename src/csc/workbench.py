import os
import time
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Union

from .dictionary import ZSDictionary
from .atoms import ZSAtom
from .utils import *

class OMPWorkbench:

    def __init__(self, signals_path:str, omp_path: str):

        # Signals data
        self.signals_path = signals_path
        self.signals_data = None

        # OMP outputs data
        self.omp_path = omp_path
        self.omp_data = None

        # Workbench state
        self.loaded = False

        # ZSDictionary
        self.dictionary = None

    def load_data(self):
        """
        Charge data from the specified file path using pandas.
        """
        with open(self.db_path, 'r') as f:
            self.data = json.load(f)
            self.signals_length = self.data['signalLength']
            self.batch_size = self.data['batchSize']
            self.snr_levels = self.data['snrLevels']
            self.sparsity_levels = self.data['sparsityLevels']
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
    
    def approxDictFromId(self, id:int) -> Dict:
        """
        Get the approximation dictionary from its ID.
        Args:
            id (int): Approximation ID.
        Returns:
            dict: Approximation dictionary.
        """
        if not self.loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        dict_approx = next((item for item in self.omp_data['omp'] if item['id'] == id), None)
        return dict_approx

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
        for true_atom in true_atoms:
            for approx_atom in approx_atoms:
                if true_atom.isEqual(approx_atom):
                    matched_atoms.append((true_atom, approx_atom))
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
        return [abs(true_atom['x'] - approx_atom['x']) for true_atom, approx_atom in OMPWorkbench.positionMatching(true_atoms, approx_atoms)]
    
    def computePositionErrors(self) :
        pass