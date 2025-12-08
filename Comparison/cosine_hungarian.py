from matchms.similarity import CosineHungarian
from matchms import calculate_scores
from matchms.importing import load_from_mgf
from Preprocessing.spectra_preprocessing import spectra_preprocessing
def cosine_hungarian(file1,file2 = None):
    spectra1 = list(load_from_mgf(file1))
    if file2 is None:
        spectra2 = spectra1
    else:
        spectra2 = list(load_from_mgf(file2))
    spectra1 = spectra_preprocessing(spectra1)
    spectra2 = spectra_preprocessing(spectra2)
    similarity_measure = CosineHungarian()
    scores = calculate_scores(spectra1,spectra2,similarity_measure,is_symmetric=False)
    scores_array = scores.scores.to_array()
    return scores_array["CosineHungarian_score"]