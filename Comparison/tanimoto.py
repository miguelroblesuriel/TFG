import numpy as np
def tanimoto_comparison(mz1,mz2):
    matches = 0
    for a in range(len(mz1)):
        for b in range(len(mz2)):
            if mz1[a] == mz2[b]:
                matches = matches + 1
    score = matches/(len(mz1) + len(mz2) - matches)
    return score

def tanimoto(spectra1,spectra2):
    n1, n2 = len(spectra1), len(spectra2)
    scores = np.zeros((n1, n2), dtype=float)
    for i in range(len(spectra1)):
        for j in range(len(spectra2)):
            scores[i][j] = tanimoto_comparison(spectra1[i].mz,spectra2[j].mz)
    return scores
