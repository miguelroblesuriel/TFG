import numpy as np
def create_embeddings(scan,intensity,mz,precmz):
    embedding = []
    embedding.append(precmz.tolist())
    embedding.append(mz.tolist())
    embedding.append(intensity.tolist())
    embedding= np.concatenate(embedding).tolist()
    scan_info = {
        'scan': scan,
        'embedding': embedding
    }
    return scan_info, len(embedding)