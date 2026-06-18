from Model_training.NeuralNetwork import NeuralNetwork
import torch

def importar_modelo(filename):
    modelo = NeuralNetwork(202)
    pesos = torch.load(filename, map_location=torch.device('cpu'))
    modelo.load_state_dict(pesos)
    modelo.eval()
    return modelo