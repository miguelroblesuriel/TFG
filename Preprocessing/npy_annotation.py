import os

from massql import msql_fileloading
from psims.test.utils import output_path

from Preprocessing.get_instrument_type import get_instrument_type
from Preprocessing.check_intensity import check_dupla_intensity
from Preprocessing.get_CE import get_CE
import numpy as np

input_filenames = os.listdir("/mnt/d/triplet_data5/")
for file_name in input_filenames:
    if file_name.endswith(".npy"):
        npy_file_path = "/mnt/d/triplet_data5/" + file_name
        mzml_file_name = file_name.replace("_triplets.npy",".mzML" )
        mzml_file_path = "/mnt/d/filedownloads_flat/" + mzml_file_name
        output_filename = file_name.replace("_triplets.npy", "_triplets_anotado.npy")
        output_path = "/mnt/d/npy_anotados/" + output_filename
        if not os.path.exists(output_path):
            ms1_df, ms2_df = msql_fileloading.load_data(mzml_file_path, cache='feather')
            loaded_data = np.load(npy_file_path, allow_pickle=True)
            scans = []
            dicts = []
            for datos in loaded_data:
                if datos["triplet"]:
                    dupla = datos["dupla"].tolist()
                    i = check_dupla_intensity(ms2_df, dupla)
                    scans.append(dupla[0])
                    scans.append(dupla[1])
                    new_triplet_dict = {
                        'dupla': datos["dupla"],
                        'triplet': datos["triplet"],
                        'scores' : datos["scores"],
                        'dupla_score' : datos["dupla_score"],
                        'score_check' : datos["score_check"],
                        'rt_check' : datos["rt_check"],
                        'dupla_i' : i
                    }
                    dicts.append(new_triplet_dict)

            try:
                ce = get_CE(mzml_file_path,scans)
                instrument_name, analyzers = get_instrument_type(mzml_file_path)
                print(ce)
                print(instrument_name)
                print(analyzers)
                datos_finales = {
                    'diccionarios': dicts,
                    'ce': ce,
                    'instrument_name': instrument_name,
                    'analyzers': analyzers,
                }

                np.save(output_path, datos_finales)
            except Exception as e:
                print("Error leyendo:", e)