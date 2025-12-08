import numpy as np
import pandas as pd
from massql import msql_fileloading
from Model_training.create_embeddings import create_embeddings
from Model_training.add_padding import add_padding
import sqlite3
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
            for i in range(3):
                if i in idx:
                    scans.append(triplet['dupla'][i])
            if triplet['triplet'] != []:
                for number in triplet['triplet']:
                    scans.append(number)

        embedded_scans =[]
        unique_scans = []
        max_length = 0
        for scan in scans:
            if scan not in unique_scans:
                unique_scans.append(scan)
                embedding, length = create_embeddings(scan,ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy(),(np.sort(ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy())), ms2_df[ms2_df['scan'] == scan]['precmz'].unique())
                embedded_scans.append(embedding)
                if length > max_length:
                    max_length = length

        conn = sqlite3.connect("D:/embeddings.db")
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            scan INTEGER NOT NULL,
            filename TEXT NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY (scan, filename)
        )
        """)

        conn.commit()


        for embedding in embedded_scans:
            embedding['embedding']  = add_padding(embedding['embedding'], max_length)
            scan = embedding['scan']
            embedding = embedding['embedding']
            cursor.execute("""
            INSERT OR REPLACE INTO embeddings (scan, filename, embedding)
            VALUES (?, ?, ?)
            """, (scan, filename, json.dumps(embedding)))

            conn.commit()

        conn.close()
        np.save('unique_embedded_scans.npy', embedded_scans)
    except Exception as e:
        print(f"An error occurred while processing {filename}: {e}")


if __name__ == "__main__":
    input_npy_filenames = [f for f in os.listdir('D:/triplet_data') if f.endswith(".npy")]
    input_mzml_filenames = [f for f in os.listdir('D:/filedownloads_flat') if f.endswith(".mzML")]
    for npy_filename in input_npy_filenames:
        filename = npy_filename.replace("_triplets.npy", "")
        npy_fileroute = os.path.join('D:/triplet_data', npy_filename)
        mzml_fileroute = os.path.join('D:/filedownloads_flat', filename + ".mzML")
        if os.path.exists(mzml_fileroute):
            print("analysing file: ", filename)
            extract_embeddings(npy_fileroute, mzml_fileroute, filename)