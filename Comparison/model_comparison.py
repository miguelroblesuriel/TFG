from Comparison.obtain_model_distance import obtain_model_distance
import math
import torch

def model_comparison(spectra1, spectra2, model):
    comparion_table = []
    for spec1 in spectra1:
        spec1 = torch.tensor(spec1, dtype=torch.float32)
        print(spec1)
        vector1 =model([spec1])
        comparison_row = []
        for spec2 in spectra2:
            spec2 = torch.tensor(spec2, dtype=torch.float32)
            vector2 = model([spec2])
            distancia = obtain_model_distance(vector1, vector2)
            similitud = math.exp(-distancia)
            comparison_row.append(similitud)
        comparion_table.append(comparison_row)

    return comparion_table

