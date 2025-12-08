import numpy as np
from matchms.importing import load_from_mgf
from matchms.filtering import add_fingerprint
from matchms.similarity import FingerprintSimilarity
from matchms import calculate_scores

def tanimoto(file1,file2 = None):
    spectra1 = list(load_from_mgf(file1))
    if file2 is None:
        spectra2 = spectra1
    else:
        spectra2 = list(load_from_mgf(file2))
    spectra1 = [add_fingerprint(s) for s in spectra1]
    spectra2 = [add_fingerprint(s) for s in spectra2]
    similarity_measure = FingerprintSimilarity()
    scores = calculate_scores(spectra1, spectra2, similarity_measure, is_symmetric=False)
    scores_array = scores.scores.to_array()
    return scores_array
