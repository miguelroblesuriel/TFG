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
from matchms.importing import load_from_mgf
from pathlib import Path

def extract_embeddings(path_data,filename):
    try:
        file_mgf = os.path.join(path_data,
                                filename)

        spectra = list(load_from_mgf(file_mgf))
        i = 1
        max_length = 0
        embedded_scans = []
        for spec in spectra:
            if len(spec.intensities) > 0:
                intensidades = spec.intensities
                intensidades_normalized = intensidades / np.max(intensidades)
                embedding, length = create_embeddings_2(i,intensidades_normalized,spec.mz, spec.metadata['precursor_mz'])
                embedded_scans.append(embedding)
                if length > max_length:
                    max_length = length
            i += 1


        conn = psycopg2.connect(
            dbname="postgres",
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
                        INSERT INTO embeddings_2 (scan, filename, embedding)
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
    path_data = "./"
    filename = "GNPS-CANDIDATE-CARNITINES-MASSQL_cleaned.mgf"
    extract_embeddings(path_data,filename)