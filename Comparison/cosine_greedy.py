import numpy as np
from matchms.similarity import CosineGreedy
from matchms import calculate_scores
def cosine_greedy(spectra1,spectra2):
    similarity_measure = CosineGreedy()
    scores = calculate_scores(spectra1,spectra2,similarity_measure,is_symmetric=False)
    scores_array = scores.scores.to_array()
    return scores_array["CosineGreedy_score"]