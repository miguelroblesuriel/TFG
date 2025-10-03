from ms2deepscore.models import load_model
import numpy as np
from ms2deepscore import MS2DeepScore
def ms2DeepScore(spectra1,spectra2):
    model_file_name = "ms2deepscore_model.pt"
    model = load_model(model_file_name)

    ms2ds = MS2DeepScore(model)

    embeddings1 = ms2ds.get_embedding_array(spectra1)
    embeddings2 = ms2ds.get_embedding_array(spectra2)

    emb1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1)[:, None]
    emb2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1)[:, None]

    similarity_matrix = np.matmul(emb1_norm, emb2_norm.T)
    return similarity_matrix