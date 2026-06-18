import numpy as np
def create_embeddings(scan,intensity,mz,precmz):
    embedding = (
            [precmz] +                                            #precmz.astype(float).tolist()
            mz.astype(float).tolist() + [-1] +
            intensity.astype(float).tolist()
    )
    scan_info = {
        'scan': scan,
        'embedding': embedding
    }
    return scan_info, len(embedding)