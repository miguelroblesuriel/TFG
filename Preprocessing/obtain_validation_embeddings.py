from Model_training.create_embeddings_2 import create_embeddings_2
import numpy as np
from Model_training.add_padding_2 import add_padding_2
def obtain_validation_embeddings(spectra):
    embeddings = []
    for spectrum in spectra:
        mz_array = spectrum.peaks.mz
        intensity_array = spectrum.peaks.intensities
        precursor_mz = spectrum.get("precursor_mz")
        indices_top = np.argsort(intensity_array)[-100:]
        indices_ordenados = np.sort(indices_top)
        i_filtrado = intensity_array[indices_ordenados]
        mz_filtrado = mz_array[indices_ordenados]
        emb, length = create_embeddings_2(0, i_filtrado, mz_filtrado, precursor_mz)
        emb = emb['embedding']
        emb = add_padding_2(emb, 100)
        masc = []
        for element in emb[0]:
            if element:
                masc.append(0)
            else:
                masc.append(1)
        emb.append(masc)
        embeddings.append(emb)
    return embeddings