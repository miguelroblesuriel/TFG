import numpy as np
import os
import random
import shutil


def count_triplets(filepath):
    loaded_data = np.load(filepath, allow_pickle=True)
    real_data = loaded_data.item()
    diccionarios = real_data["diccionarios"]
    contadordetripletas = 0
    tripletas = 0
    for item in diccionarios:
        if item["triplet"] != []:
            if len(item["triplet"]) > 1:
                contadordetripletas = contadordetripletas + len(item["triplet"])
                tripletas = tripletas + 1

    return contadordetripletas, tripletas


if __name__ == "__main__":
    input_filenames = os.listdir("/mnt/d/npy_anotados/")
    path_training = "/mnt/d/npy_anotados_training_2/"
    path_testing = "/mnt/d/npy_anotados_testing_2/"
    contadordetripletas = 0
    tripletas = 0
    for file_name in input_filenames:
        if  file_name.endswith(".npy"):
            file_path = "/mnt/d/npy_anotados/" + file_name
            print(file_name)
            contador1, triplet1 = count_triplets(file_path)
            contadordetripletas = contadordetripletas + contador1
            tripletas = triplet1 + tripletas


    print(contadordetripletas)
    print(tripletas)
    tripletas_acumuladas = 0
    carpeta_training = []
    carpeta_testing = []
    random.shuffle(input_filenames)
    for file_name in input_filenames:
        if  file_name.endswith(".npy"):
            file_path = "/mnt/d/npy_anotados/" + file_name
            contador1, triplet1 = count_triplets(file_path)
            if (contador1 + tripletas_acumuladas) < contadordetripletas*0.8:
                shutil.copy(file_path, os.path.join(path_training, file_name))
                carpeta_training.append(file_name)
                tripletas_acumuladas = tripletas_acumuladas + contador1
            else:
                shutil.copy(file_path, os.path.join(path_testing, file_name))
                carpeta_testing.append(file_name)
    print(carpeta_training)
