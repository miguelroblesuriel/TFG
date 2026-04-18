
def filtro_npy(condiciones,diccionario):
    if  diccionario['ce'] in condiciones['ce'] and condiciones['ce'] != []:
        return False
    if  diccionario['instrument_name'] not in condiciones['instrument_name'] and condiciones['instrument_name'] != []:
        return False
    if diccionario['analyzers'] not in condiciones['analyzers'] and condiciones['analyzers'] != []:
        return False
    new_data = []
    for dic in diccionario['diccionarios']:
        if dic['dupla_i'] >= condiciones['dupla_i']:
            new_data.append(dic)
    return True, new_data

