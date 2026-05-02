from matchms.similarity import ModifiedCosine
from matchms import calculate_scores
from matchms.importing import load_from_mgf
from Preprocessing.spectra_preprocessing import spectra_preprocessing
def modified_cosine(spectra1,spectra2 = None):
    if spectra2 is None:
        spectra2 = spectra1
    similarity_measure = ModifiedCosine()
    scores = calculate_scores(spectra1,spectra2,similarity_measure,is_symmetric=False)
    scores_array = scores.scores.to_array()
    return scores_array["ModifiedCosine_score"], scores