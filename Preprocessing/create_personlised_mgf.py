from pyteomics import mgf
from matchms.importing import load_from_mgf
import random

# 1. Configuración
archivo_original = "GNPS-SELLECKCHEM-FDA-PART1_filtrado.mgf"
archivo_nuevo = "GNPS-SELLECKCHEM-FDA-PART1_filtrado.mgf"
spectra = list(load_from_mgf(archivo_original))
instrumentos_aceptados = ['LC-ESI-qTof', 'N/A-ESI-QTOF']
with open("inchis_comunes.txt", "r") as archivo:
    espectros_seleccionados = []
    inchis_buscados = {linea.strip() for linea in archivo if linea.strip()}
    for inchi in inchis_buscados:
        for espectro in spectra:
            titulo_actual = espectro.metadata.get('inchi')
            instrumento_actual = espectro.metadata.get('instrument_type')
            if titulo_actual == inchi and instrumento_actual in instrumentos_aceptados:
                print("Espectro seleccionado")
                espectros_seleccionados.append(espectro)
                break

print(len(espectros_seleccionados))
"""
candidatos_aleatorios = []
for spec in spectra:
    inchi_actual = spec.metadata.get('inchi')
    inst_actual = spec.metadata.get('instrument_type')  # O la clave que confirmamos antes

    # Condición: Instrumento correcto Y InChI no repetido
    if inst_actual in instrumentos_aceptados and inchi_actual not in inchis_buscados:
        candidatos_aleatorios.append(spec)
espectros_aleatorios = random.sample(candidatos_aleatorios, 936)
for espectro in espectros_aleatorios:
    espectros_seleccionados.append(espectro)
"""
espectros_para_guardar = []
for spec in espectros_seleccionados:
    nuevo_dict = {
        'm/z array': spec.peaks.mz,
        'intensity array': spec.peaks.intensities,
        'params': spec.metadata.copy()
    }
    espectros_para_guardar.append(nuevo_dict)

mgf.write(espectros_para_guardar, output=archivo_nuevo)

print(f"Se han guardado {len(espectros_para_guardar)} espectros en {archivo_nuevo}")