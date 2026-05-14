import numpy as np
import pandas as pd
import subprocess
from massql import msql_fileloading
from Model_training.create_embeddings import create_embeddings
from Model_training.create_embeddings_2 import create_embeddings_2
from Model_training.add_padding import add_padding
from Model_training.add_padding_2 import add_padding_2
import psycopg2
import json
import os
from pathlib import Path
def extract_embeddings(npy_fileroute,mzml_fileroute,filename):
    try:
        data = np.load(npy_fileroute, allow_pickle= True)
        ms1_df, ms2_df = msql_fileloading.load_data(mzml_fileroute)
        scans = []
        for triplet in data:
            idx = triplet['dupla'].index
            print(idx)
            for i in range(idx.max()+1):
                if i in idx:
                    scans.append(triplet['dupla'][i])
                    print(triplet['dupla'][i])
            if triplet['triplet'] != []:
                for number in triplet['triplet']:
                    scans.append(number)

        embedded_scans =[]
        unique_scans = []
        max_length = 0
        for scan in scans:
            if scan not in unique_scans:
                unique_scans.append(scan)
                i = ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy()
                mz = ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy()
                indices_top = np.argsort(i)[-100:]
                indices_ordenados = np.sort(indices_top)
                i_filtrado = i[indices_ordenados]
                mz_filtrado = mz[indices_ordenados]
                embedding, length = create_embeddings_2(scan,i_filtrado,mz_filtrado, ms2_df[ms2_df['scan'] == scan]['precmz'].unique())
                embedded_scans.append(embedding)
                if length > max_length:
                    max_length = length


        conn = psycopg2.connect(
            dbname="embeddings",
            user="postgres",
            password="postgres",
            host='172.25.128.1',
            port="5432"
        )
        cursor = conn.cursor()

        try:
            for embedding_item in embedded_scans:
                padded_emb = add_padding_2(embedding_item['embedding'], max_length)
                print(padded_emb)
                scan_val = int(embedding_item['scan'])
                print(scan_val)
                embedding_bytes = json.dumps(padded_emb).encode('utf-8')

                query = """
                        INSERT INTO embeddings (scan, filename, embedding)
                        VALUES (%s, %s, %s) ON CONFLICT (scan, filename) 
                DO \
                        UPDATE SET embedding = EXCLUDED.embedding; \
                        """

                cursor.execute(query, (scan_val, filename, embedding_bytes))

            conn.commit()
            print("Datos insertados/actualizados correctamente.")

        except Exception as e:
            conn.rollback()
            print(f"Error al insertar: {e}")

        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        print(f"An error occurred while processing {filename}: {e}")


if __name__ == "__main__":
    input_npy_filenames = [f for f in os.listdir('/mnt/d/triplet_data5/') if f.endswith(".npy")]
    input_mzml_filenames = [f for f in os.listdir('/mnt/d/filedownloads_flat/') if f.endswith(".mzML")]
    for npy_filename in input_npy_filenames:
        filename = npy_filename.replace("_triplets.npy", "")
        npy_fileroute = os.path.join('/mnt/d/triplet_data5/', npy_filename)
        mzml_fileroute = os.path.join('/mnt/d/filedownloads_flat/', filename + ".mzML")
        if os.path.exists(mzml_fileroute):
            print("analysing file: ", filename)
            extract_embeddings(npy_fileroute, mzml_fileroute, filename)