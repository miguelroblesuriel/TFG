import torch
def obtain_model_distance(v1, v2):

    diferencia_al_cuadrado = (v1 - v2) ** 2

    # 2. Sumamos las diferencias y sumamos el eps (como hace PyTorch internamente)
    suma_cuadrados = torch.sum(diferencia_al_cuadrado, dim=1) + 1e-7

    # 3. Aplicamos la raíz cuadrada (ya que p=2)
    distancia = torch.sqrt(suma_cuadrados)

    print("Distancia exacta:", distancia.item())
    return distancia