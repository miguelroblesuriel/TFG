import numpy as np
from matchms.filtering import default_filters, normalize_intensities, add_fingerprint
from matchms.similarity import FingerprintSimilarity, CosineGreedy, ModifiedCosine, CosineHungarian
from matchms.importing import load_from_mgf
from matchms import calculate_scores

from Comparison.get_hits_standard import get_hits_standard
from Comparison.modified_cosine import modified_cosine
from Comparison.cosine_greedy import cosine_greedy
from Comparison.get_greedy_hits import get_greedy_hits
from Comparison.get_modified_hits import get_modified_hits
from Comparison.cosine_hungarian import cosine_hungarian
from Comparison.tanimoto import tanimoto
from Comparison.ms2DeepScore import ms2DeepScore_standard, ms2DeepScore_trained
from Comparison.extensive_method_difference import  extensive_method_difference
from Preprocessing.identify_spectra import identify_spectra
from Visualization.plot_differences_histogram import plot_differences_histogram
from Visualization.dot_plot import dot_plot
from Preprocessing.spectra_preprocessing import spectra_preprocessing
def obtener_espectros_en_comun(spectra1,spectra2):
    inchis = []
    for spec in spectra1:
        inchi1 = spec.metadata.get('inchi')
        if inchi1:
            for spec2 in spectra2:
                inchi2 = spec2.metadata.get('inchi')
                if inchi1 == inchi2 and inchi1 not in inchis:
                    inchis.append(inchi1)
                    print(spec.metadata['inchi'])
    return inchis

import os

path_data = "./"
file_mgf1 = os.path.join(path_data,
                        "GNPS-SELLECKCHEM-FDA-PART1_filtrado.mgf")
file_mgf2 = os.path.join(path_data,
                        "MONA_filtrado.mgf")
spectra1 = list(load_from_mgf(file_mgf1))
print("cargando archivo 2")
spectra2 = list(load_from_mgf(file_mgf2))
print(len(spectra2))
print(len(spectra1))
analyzers_64 = []
analyzers_1000 = []
for spec in spectra1:
    if spec.metadata.get('instrument_type') not in analyzers_64:
        analyzers_64.append(spec.metadata['instrument_type'])
for spec in spectra2:
    if spec.metadata.get('instrument_type') not in analyzers_1000:
        analyzers_1000.append(spec.metadata['instrument_type'])
print(analyzers_64)
print(analyzers_1000)
modified_scores, modified_scores_raw = modified_cosine(spectra2, spectra1)
hits, mean = get_modified_hits(spectra1, modified_scores_raw)
print(hits)
print(mean)
print("-------------------------------------------------------------------------------")
modified_scores, modified_scores_raw = modified_cosine(spectra1, spectra2)
comparison_table = identify_spectra(spectra1, spectra2)
hits,mean = get_hits_standard(modified_scores, comparison_table)
print(hits)
print(mean)
print("-------------------------------------------------------------------------------")
greedy_scores, greedy_scores_raw = cosine_greedy(spectra2,spectra1)
hits, mean = get_greedy_hits(spectra1, greedy_scores_raw)
print(hits)
print(mean)
print("-------------------------------------------------------------------------------")
greedy_scores, greedy_scores_raw = cosine_greedy(spectra1,spectra2)
hits,mean = get_hits_standard(greedy_scores, comparison_table)
print(hits)
print(mean)
print("-------------------------------------------------------------------------------")
"""
tanimoto_scores = tanimoto(file_mgf)


hungarian_scores = cosine_hungarian(file_mgf)
ms2_standard_scores = ms2DeepScore_standard(file_mgf)
"""
"""
print("Ms2: ")
extensive_method_difference(tanimoto_scores,ms2_standard_scores)

print("Greedy: ")
extensive_method_difference(tanimoto_scores,greedy_scores)

print("Modified: ")
extensive_method_difference(tanimoto_scores,modified_scores)
print("Hungarian: ")
extensive_method_difference(tanimoto_scores,hungarian_scores)
"""

"""
dot_plot(tanimoto_scores,ms2_standard_scores)
dot_plot(tanimoto_scores,hungarian_scores)

dot_plot(tanimoto_scores,greedy_scores)

dot_plot(tanimoto_scores,modified_scores)
"""


"""
with open("resultados.txt", "w", encoding="utf-8") as f:
    f.write(f"Ms2: {method_difference(tanimoto_scores, ms2_scores)}\n")
    f.write(f"Greedy: {method_difference(tanimoto_scores, greedy_scores)}\n")
    f.write(f"Modified: {method_difference(tanimoto_scores, modified_scores)}\n")
    f.write(f"Hungarian: {method_difference(tanimoto_scores, hungarian_scores)}\n")
    


scores= []
scores.append(greedy_scores)
scores.append(modified_scores)
scores.append(ms2_scores)
plot_differences_histogram(tanimoto_scores,scores)
"""
