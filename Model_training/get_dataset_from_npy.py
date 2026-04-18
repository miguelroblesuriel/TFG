import numpy as np
import json
import os
from Model_training.Spectra_dataset import Spectra_dataset
import psycopg2


def obtener_embedding_individual(conn, nombre_archivo, scan_id):
    """
    Recupera el embedding de un archivo para un scan específico.
    """

    query = """
            SELECT embedding
            FROM embeddings
            WHERE filename = %s \
              AND scan = %s; \
            """

    with conn.cursor() as cursor:
        cursor.execute(query, (nombre_archivo, scan_id))
        resultado = cursor.fetchone()

        if resultado:
            return resultado[0]
        else:
            return None

def get_dataset_from_npy(nombre_archivo, filepath):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host='172.25.128.1',
        port="5432"
    )
    database_filename = nombre_archivo
    file_name = os.path.join(nombre_archivo + '_triplets_anotado')
    file_path = os.path.join(filepath + nombre_archivo +'_triplets_anotado.npy')
    loaded_data = np.load(file_path, allow_pickle=True)  # importar los sets de dupla/tripletas
    real_data = loaded_data.item()
    dict = real_data['diccionarios']
    duplas = []
    scores = []
    datasets = []
    contadordetripletas = 0
    for item in dict:
        triplets = []
        if item["triplet"] != []:
            if len(item["triplet"]) > 1:
                contadordetripletas = contadordetripletas + len(item["triplet"])

            duplas = item["dupla"].tolist()
            embedding_dupla1 = obtener_embedding_individual(conn, database_filename, duplas[0])
            embedding_dupla2 = obtener_embedding_individual(conn, database_filename, duplas[1])
            embedding_dupla1 = embedding_dupla1.tobytes().decode('utf-8')
            embedding_dupla2 = embedding_dupla2.tobytes().decode('utf-8')
            embedding_dupla1 = json.loads(embedding_dupla1)
            embedding_dupla2 = json.loads(embedding_dupla2)
            duplas = [embedding_dupla1, embedding_dupla2]
            for triplet in item["triplet"]:
                embedding_triplet = obtener_embedding_individual(conn, database_filename, int(triplet))
                embedding_triplet = embedding_triplet.tobytes().decode('utf-8')
                embedding_triplet = json.loads(embedding_triplet)
                triplets.append(embedding_triplet)
            datasets.append(Spectra_dataset(duplas, triplets, item["scores"]))
    return datasets

if __name__ == "__main__":
    nombre_archivo = '00000000poole_4_01_14268'
    print(get_dataset_from_npy(nombre_archivo))