from Model_training.NeuralNetwork_Complex import NeuralNetwork_Complex
import torch

def importar_modelo_2(filename):
    modelo = NeuralNetwork_Complex(100)
    pesos = torch.load(filename, map_location=torch.device('cpu'))
    modelo.load_state_dict(pesos)
    modelo.eval()
    return modelo


if __name__ == "__main__":
    mi_modelo = importar_modelo_2("pesos_modelo512_0.6.pt")
    for nombre_capa, tensores in mi_modelo.state_dict().items():
        # Convertimos a float para asegurar que calcula la media sin problemas
        pesos_float = tensores.float()
        print(
            f"{nombre_capa} | Media: {pesos_float.mean().item():.4f} | Desviación Estándar: {pesos_float.std().item():.4f}")
