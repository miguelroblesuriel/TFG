import numpy as np
from matchms.filtering import default_filters, normalize_intensities, add_fingerprint
from matchms.similarity import FingerprintSimilarity, CosineGreedy, ModifiedCosine, CosineHungarian
from matchms.importing import load_from_mgf
from matchms import calculate_scores
from Comparison.modified_cosine import modified_cosine
from Comparison.cosine_greedy import cosine_greedy
from Comparison.tanimoto import tanimoto
from Comparison.ms2DeepScore import ms2DeepScore
from Comparison.method_difference import method_difference
from Visualization.plot_differences_histogram import plot_differences_histogram

import os

path_data = "./"
file_mgf = os.path.join(path_data,
                        "datos.mgf")
spectra = list(load_from_mgf(file_mgf))
processed_spectra = []
for s in spectra:
    s = default_filters(s)           # cleans peaks, removes empty peaks
    s = normalize_intensities(s)     # normalize intensities to 0-1
    s = add_fingerprint(s)           # add fingerprint
    processed_spectra.append(s)

ms2_scores = ms2DeepScore(processed_spectra,processed_spectra)
greedy_scores = cosine_greedy(processed_spectra,processed_spectra)
modified_scores = modified_cosine(processed_spectra,processed_spectra)
tanimoto_scores = tanimoto(processed_spectra,processed_spectra)

"""
print("Ms2: ", method_difference(tanimoto_scores,ms2_scores))
print("Greedy: ", method_difference(tanimoto_scores,greedy_scores))
print("Modified: ", method_difference(tanimoto_scores,modified_scores))
"""
scores= []
scores.append(greedy_scores)
scores.append(modified_scores)
scores.append(ms2_scores)
plot_differences_histogram(tanimoto_scores,scores)


