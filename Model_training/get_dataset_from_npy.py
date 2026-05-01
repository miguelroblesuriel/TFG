import numpy as np
import json
import os
from Model_training.Spectra_dataset import Spectra_dataset
import psycopg2


def obtener_embedding_archivo(conn, nombre_archivo):
    """
        Recupera miles de pares scan/embedding para un solo archivo
        y los organiza en un mapa.
        """
    query = """
            SELECT scan, embedding
            FROM embeddings_2
            WHERE filename = %s; \
            """

    mapa_resultado = {}

    try:
        # Usamos un nombre de cursor para activar un "Server-side cursor"
        # si los datos son realmente masivos (opcional en la mayoría de casos)
        with conn.cursor() as cursor:
            print(f"Buscando datos para: {nombre_archivo}...")
            cursor.execute(query, (nombre_archivo,))

            # Iterar directamente sobre el cursor es más eficiente que fetchall()
            count = 0
            for scan_id, embedding in cursor:
                mapa_resultado[scan_id] = embedding
                count += 1

            if count > 0:
                print(f"Éxito: Mapa creado con {count} pares para el archivo.")
            else:
                print("No se encontraron datos.")

            return mapa_resultado

    except Exception as e:
        print(f"Error procesando miles de registros: {e}")
        return {}

def obtener_embedding_individual(conn, nombre_archivo, scan_id):
    """
    Recupera el embedding de un archivo para un scan específico.
    """

    query = """
            SELECT embedding
            FROM embeddings_2
            WHERE filename = %s \
              AND scan = %s; \
            """

    with conn.cursor() as cursor:
        #print("Solicitud base de datos")
        cursor.execute(query, (nombre_archivo, scan_id))
        resultado = cursor.fetchone()
        #print("Respuesta base de datos")
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
    mapa = obtener_embedding_archivo(conn, database_filename)
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
            embedding_dupla1 = mapa[duplas[0]]
            embedding_dupla2 = mapa[duplas[1]]
            embedding_dupla1 = embedding_dupla1.tobytes().decode('utf-8')
            embedding_dupla2 = embedding_dupla2.tobytes().decode('utf-8')
            embedding_dupla1 = json.loads(embedding_dupla1)
            embedding_dupla2 = json.loads(embedding_dupla2)
            duplas = [embedding_dupla1, embedding_dupla2]
            for triplet in item["triplet"]:
                embedding_triplet = mapa[int(triplet)]
                embedding_triplet = embedding_triplet.tobytes().decode('utf-8')
                embedding_triplet = json.loads(embedding_triplet)
                triplets.append(embedding_triplet)
            datasets.append(Spectra_dataset(duplas, triplets, item["scores"]))
    return datasets

if __name__ == "__main__":
    nombre_archivo = '00000000poole_4_01_14268'
    print(get_dataset_from_npy(nombre_archivo))