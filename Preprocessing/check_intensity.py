from massql import msql_fileloading
import numpy as np

def check_dupla_intensity(ms2_df, dupla):
    i_min = 0
    for scan in dupla:
        if i_min == 0:
            i_min = ms2_df.loc[ms2_df['scan'] == scan]['i'].max()
        elif i_min > ms2_df.loc[ms2_df['scan'] == scan]['i'].max():
            i_min = ms2_df.loc[ms2_df['scan'] == scan]['i'].max()
        if (ms2_df.loc[ms2_df['scan'] == scan]['i'].max() < 10e3):
            print("intensidad baja")
    return i_min


if __name__ == '__main__':
    file_path = '/mnt/d/triplet_data5/20190429_JJ_VB_BioSFA_Pttime_1_QE144_Ag68377-924_USHXG01160_NEG_MSMS-v2_1_extr-ctrl--extr-ctrl-_1_IR02_NA_083_triplets.npy'
    mzml_path = '/mnt/d/filedownloads_flat_1/20190429_JJ_VB_BioSFA_Pttime_1_QE144_Ag68377-924_USHXG01160_NEG_MSMS-v2_1_extr-ctrl--extr-ctrl-_1_IR02_NA_083.mzML'
    loaded_data = np.load(file_path, allow_pickle=True)
    duplas = [datos["dupla"].tolist() for datos in loaded_data]
    ms1_df, ms2_df = msql_fileloading.load_data(mzml_path, cache='feather')
    for dupla in duplas:
        print(check_dupla_intensity(ms2_df, dupla))