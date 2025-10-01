import numpy as np
from matchms.filtering import default_filters, normalize_intensities, add_fingerprint
from matchms.similarity import FingerprintSimilarity, CosineGreedy, ModifiedCosine, CosineHungarian
from matchms.importing import load_from_mgf
from matchms import calculate_scores
from Comparison.modified_cosine import modified_cosine
from Comparison.cosine_greedy import cosine_greedy
from Comparison.tanimoto import tanimoto
from Comparison.ms2DeepScore import ms2DeepScore

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

print(cosine_greedy(processed_spectra,processed_spectra))
print(modified_cosine(processed_spectra,processed_spectra))
print(tanimoto(processed_spectra,processed_spectra))

