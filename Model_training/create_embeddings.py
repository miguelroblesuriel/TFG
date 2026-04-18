import numpy as np
def create_embeddings(scan,intensity,mz,precmz):
    embedding = (
            precmz.astype(float).tolist() +
            mz.astype(float).tolist() +
            intensity.astype(float).tolist()
    )
    scan_info = {
        'scan': scan,
        'embedding': embedding
    }
    return scan_info, len(embedding)