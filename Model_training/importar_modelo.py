from Model_training.NeuralNetwork_Complex import NeuralNetwork_Complex
import torch

def importar_modelo(filename):
    modelo = NeuralNetwork_Complex(100)
    pesos = torch.load(filename, map_location=torch.device('cpu'))
    modelo.load_state_dict(pesos)
    modelo.eval()
    return modelo