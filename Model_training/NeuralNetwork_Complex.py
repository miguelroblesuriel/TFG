from numba.core.types import float16
from torch import nn
import torch.nn.functional as F
import torch
import math

from torch.nn import TransformerEncoderLayer, TransformerEncoder


class MultiFeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiFeedForwardNetwork, self).__init__()
        self._layers = []
        layer_dims = [input_size] + hidden_size + [output_size]
        for i in range(1, len(layer_dims) - 1):
            self._layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            self._layers.append(nn.ReLU())
            self._layers.append(nn.Dropout(0.4))

        self._layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self._layers = nn.Sequential(*self._layers)

    def forward(self, x):
        return  self._layers(x)

class NeuralNetwork_Complex(nn.Module):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length
        self.red_sinusoidal = MultiFeedForwardNetwork(1024, [1024], 1024)
        self.red_intensidad = MultiFeedForwardNetwork(1025, [1024], 1024)
        self.red_final = MultiFeedForwardNetwork(1024, [1024], 512)
        encoder_layer = TransformerEncoderLayer(
            1024,
            8,
            dim_feedforward=1024,
            dropout=True,
            activation = 'relu',
            batch_first=True,
            norm_first=True
        )
        self._encoder = TransformerEncoder(
            encoder_layer,
            3,
            enable_nested_tensor=False
        )

    def sinusoidal(self, lambda_min, lambda_max):
        lambda_div_value = lambda_max / lambda_min
        x = torch.arange(0, 1024, 2)
        x = (
                2 * math.pi *
                (
                        lambda_min *
                        lambda_div_value ** (x / (512 - 2))
                ) ** -1
        )
        return x


    def forward(self, x):
        print(self.max_length)
        mz_frequencies =  self.sinusoidal(math.pow(10, -3.0), math.pow(10, 3.0))
        print(mz_frequencies.shape)
        mz_transform = torch.einsum('bl,d->bld', x[0][0].unsqueeze(0), mz_frequencies)
        mz_sin = torch.sin(mz_transform)
        mz_cos = torch.cos(mz_transform)
        print(mz_sin)
        print(mz_sin.shape)
        b, l, d = mz_sin.shape
        mz = torch.zeros(b, l, 2 * d, dtype= x[0][0][0].dtype, device='cpu')
        mz[:, :, ::2] = mz_sin
        mz[:, :, 1::2] = mz_cos
        print(mz.shape)
        mz = self.red_sinusoidal(mz)
        i = x[0][1]
        i = i.unsqueeze(0)
        i = torch.unsqueeze(i, dim = -1)
        embedding_completo =self.red_intensidad(
                                    torch.cat([mz, i], dim=-1))
        print(embedding_completo.shape)
        embedding_completo = self._encoder(embedding_completo)
        print(embedding_completo.shape)
        embedding_ponderado = embedding_completo.mean(dim=1)
        print(embedding_ponderado.shape)
        embedding_final = self.red_final(embedding_ponderado)
        return  embedding_final