import numpy as np
def create_embeddings_2(scan,intensity,mz,precmz):
    embedding = [None, None]
    embedding[0] = [float(precmz)] + mz.astype(float).tolist()
    embedding[1] = [2] + intensity.astype(float).tolist()
    print(embedding)
    scan_info = {
        'scan': scan,
        'embedding': embedding
    }
    return scan_info, len(embedding[0])