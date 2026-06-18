from numba.core.types import float16
from torch import nn
import torch.nn.functional as F
import torch
import math
from sklearn.preprocessing import RobustScaler
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
        self.hidden_sinusoidal = 1024
        self.output_sinusoidal = 1024
        self.hidden_intensidad = 1024
        self.output_intensidad = 1024
        self.hidden_final = 1024
        self.output_final = 512
        self.red_sinusoidal = MultiFeedForwardNetwork(1024, [self.hidden_sinusoidal], self.output_sinusoidal)
        self.red_intensidad = MultiFeedForwardNetwork(self.output_sinusoidal+1, [self.hidden_intensidad], self.output_intensidad)
        self.red_final = MultiFeedForwardNetwork(self.output_intensidad, [self.hidden_final], self.output_final)
        encoder_layer = TransformerEncoderLayer(
            1024,
            8,
            dim_feedforward=1024,
            dropout=0.3,
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
        scaler = RobustScaler()
        mz = scaler.fit_transform(x[:, 0, :])
        mz = torch.tensor(mz, dtype=torch.float32)
        i = x[:, 1, :]
        mask = x[:, 2, :].bool()
        mz_frequencies =  self.sinusoidal(math.pow(10, -3.0), math.pow(10, 3.0))
        device = mz.device
        mz_frequencies = mz_frequencies.to(device)
        #print("Mz_frequencies", mz_frequencies)
        mz_transform = torch.einsum('bl,d->bld', mz, mz_frequencies)
        #print("Mz_transform", mz_transform.shape)
        mz_sin = torch.sin(mz_transform)
        #print(mz_sin)
        mz_cos = torch.cos(mz_transform)
        #print(mz_cos)
        b, l, d = mz_sin.shape
        """
        mz = torch.zeros(b, l, 2 * d, dtype= x[0][0][0].dtype, device=mz_sin.device)
        print("MZ:", mz)
        mz = mz.to(device)
        print("MZ:", mz)
        mz[:, :, ::2] = mz_sin
        mz[:, :, 1::2] = mz_cos
        print("MZ:", mz)
        """
        mz = torch.stack((mz_sin, mz_cos), dim=-1).flatten(2)
        #print("MZ nuevo:", mz.shape)
        mz = mz.contiguous()
        mz = self.red_sinusoidal(mz)
        #print("MZ:", mz.shape)
        i = i.to(device)
        #print("i:", i.shape)
        mask = mask.to(device)
        #print("mask:", mask)
        i = torch.unsqueeze(i, dim = -1)
        #print(torch.cat([mz, i], dim=-1).shape)
        embedding_completo =self.red_intensidad(
                                    torch.cat([mz, i], dim=-1))
        #print("embedding comleto:", embedding_completo.shape)
        embedding_completo = self._encoder(embedding_completo, src_key_padding_mask=mask)
        #print("embedding comleto:", embedding_completo.shape)
        extened_mask = mask.unsqueeze(-1)

        masked_embeddings: torch.Tensor = embedding_completo * (~extened_mask)
        #print("embedding masked:", masked_embeddings)
        sum_embeddings = masked_embeddings.sum(dim=1)
        num_valid_tokens = (~mask).sum(dim=1, keepdim=True)
        embedding_ponderado = sum_embeddings / num_valid_tokens
        #print(" embedding_ponderado:", embedding_ponderado)
        embedding_final = self.red_final(embedding_ponderado)
        #print(" embedding_final:", embedding_final)
        embedding_final = F.normalize(embedding_final, p=2, dim=1)
        return  embedding_final