import numpy as np
from .atoms import CSCAtom
from .dictionary import CSCDictionary

def conv_multipath_matching_pursuit(signal:np.ndarray, dictionary:CSCDictionary, nb_atoms:int, nb_mmp_branches:int) -> list:
    # Initialize the residual signal
    residual = signal.copy()

    # Define the dissimilarity threshold
    dissimilarity = 0.4

    # Process the depth-first search
    approx, infos = dictionary.mmpdf(signal=residual,
                     sparsity_level=nb_atoms, 
                     connections_level=nb_mmp_branches,
                     dissimilarity=dissimilarity,
                     nb_branches=nb_mmp_branches,
                     verbose=False
                     )
    
    return approx, infos
    
