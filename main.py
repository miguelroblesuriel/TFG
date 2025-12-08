import numpy as np
from matchms.filtering import default_filters, normalize_intensities, add_fingerprint
from matchms.similarity import FingerprintSimilarity, CosineGreedy, ModifiedCosine, CosineHungarian
from matchms.importing import load_from_mgf
from matchms import calculate_scores
from Comparison.modified_cosine import modified_cosine
from Comparison.cosine_greedy import cosine_greedy
from Comparison.cosine_hungarian import cosine_hungarian
from Comparison.tanimoto import tanimoto
from Comparison.ms2DeepScore import ms2DeepScore_standard, ms2DeepScore_trained
from Comparison.extensive_method_difference import  extensive_method_difference
from Visualization.plot_differences_histogram import plot_differences_histogram
from Visualization.dot_plot import dot_plot

import os

path_data = "./"
file_mgf = os.path.join(path_data,
                        "GNPS-SELLECKCHEM-FDA-PART1.mgf")




greedy_scores = cosine_greedy(file_mgf)
modified_scores = modified_cosine(file_mgf)
hungarian_scores = cosine_hungarian(file_mgf)
tanimoto_scores = tanimoto(file_mgf)
ms2_standard_scores = ms2DeepScore_standard(file_mgf)

print("Ms2: ")
extensive_method_difference(tanimoto_scores,ms2_standard_scores)
print("Greedy: ")
extensive_method_difference(tanimoto_scores,greedy_scores)
print("Modified: ")
extensive_method_difference(tanimoto_scores,modified_scores)
print("Hungarian: ")
extensive_method_difference(tanimoto_scores,hungarian_scores)


dot_plot(tanimoto_scores,ms2_standard_scores)
dot_plot(tanimoto_scores,hungarian_scores)
dot_plot(tanimoto_scores,greedy_scores)
dot_plot(tanimoto_scores,modified_scores)


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
