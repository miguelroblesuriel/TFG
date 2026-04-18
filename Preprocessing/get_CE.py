from pyteomics import mzml
import numpy as np

def get_CE(mzml_path, scans):
    with mzml.read(mzml_path, use_index=True) as reader:
        ce = []
        for scan in scans:
            try:
                spectrum = reader[scan-1]
                found_values = []
                search_stack = [spectrum]
                while search_stack:
                    current = search_stack.pop()

                    if isinstance(current, dict):
                        for k, v in current.items():
                            if k == 'collision energy':
                                found_values.append(v)
                            elif isinstance(v, (dict, list)):
                                search_stack.append(v)

                    elif isinstance(current, list):
                        for item in current:
                            if isinstance(item, (dict, list)):
                                search_stack.append(item)

                if found_values:
                    for val in set(found_values):
                        if val not in ce:
                            ce.append(val)
                else:
                    print(f"Scan {scan}: No 'collision energy' key found.")

            except Exception as e:
                print(f"Error in scan {scan}: {e}")
    return ce






if __name__ == "__main__":
    file_path = '/mnt/d/triplet_data5/20190429_JJ_VB_BioSFA_Pttime_1_QE144_Ag68377-924_USHXG01160_NEG_MSMS-v2_1_extr-ctrl--extr-ctrl-_1_IR02_NA_083_triplets.npy'
    mzml_path = '/mnt/d/filedownloads_flat_1/20190429_JJ_VB_BioSFA_Pttime_1_QE144_Ag68377-924_USHXG01160_NEG_MSMS-v2_1_extr-ctrl--extr-ctrl-_1_IR02_NA_083.mzML'
    loaded_data = np.load(file_path, allow_pickle=True)
    scans = [datos["dupla"].tolist()[0] for datos in loaded_data]
    print(get_CE(mzml_path, scans))
