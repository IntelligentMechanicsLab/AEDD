"""
@author: Xiaolong HE
"""
import torch.nn as nn
import torch
from torch.autograd import Variable

#%% SiLU activation
def silu(input):
    return input * torch.sigmoid(input)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return silu(input)


#%% Multi-layer Feedforward Networks
class Net(nn.Module):
    def __init__(self, units, act_type=1):
        super(Net,self).__init__()
        self.hidden = self._make_layers1(units, act_type)
        self.fc = nn.Linear(units[-2],units[-1])
        
    def _make_layers(self, units, act_type):
        layers = []
        for i in range(len(units)-2):
            layers.append(nn.Linear(units[i],units[i+1]))
            if act_type == 1:
                layers.append(nn.ReLU())
            elif act_type == 2:
                layers.append(nn.Tanh())
            elif act_type == 3:
                layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.fc(x)
        return x

#%% Autoencoders
class AE(nn.Module):
    """
    Autoencoders with symmetric architecture for the encoder and the decoder. 
    Linear activation is applied to the embedding layer and the output layer.
    All other hidden layers have the same activation.
    
    input args:
        - n_neurons: list, contains the neuron numbers in each layer 
                of the Encoder, including the input layer.
        - act_type: int, activation for the hidden layers of the Encoder
                and the Decoder; 1: ReLU, 2: Tanh, 3: Sigmoid, 4: SiLU
    """
    def __init__(self, n_neurons, act_type=2):
        super(AE,self).__init__()
        self.encoder = self._make_layers_encoder(n_neurons, act_type)
        self.decoder = self._make_layers_decoder(n_neurons, act_type)
        
    def _make_layers_encoder(self, n_neurons, act_type):
        layers = []
        for i in range(len(n_neurons)-2):
            layers.append(nn.Linear(n_neurons[i],n_neurons[i+1]))
            if act_type == 1:
                layers.append(nn.ReLU())
            elif act_type == 2:
                layers.append(nn.Tanh())
            elif act_type == 3:
                layers.append(nn.Sigmoid())
            elif act_type == 4:
                layers.append(SiLU())
        layers.append(nn.Linear(n_neurons[-2],n_neurons[-1]))
        return nn.Sequential(*layers)
    
    def _make_layers_decoder(self, n_neurons, act_type):
        layers = []
        for i in range(len(n_neurons)-2):
            layers.append(nn.Linear(n_neurons[-1-i],n_neurons[-2-i]))
            if act_type == 1:
                layers.append(nn.ReLU())
            elif act_type == 2:
                layers.append(nn.Tanh())
            elif act_type == 3:
                layers.append(nn.Sigmoid())
            elif act_type == 4:
                layers.append(SiLU())
        layers.append(nn.Linear(n_neurons[1],n_neurons[0]))
        return nn.Sequential(*layers)
    
    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#%% Variational Autoencoders
class VAE(nn.Module):
    def __init__(self, units, act_type=2):
        super(VAE,self).__init__()
        self.encoder = self._make_layers_encoder(units, act_type)
        self.z_mean = nn.Linear(units[-2],units[-1])
        self.z_logvar = nn.Linear(units[-2],units[-1])
        self.decoder = self._make_layers_decoder(units, act_type)
        
    def _make_layers_encoder(self, units, act_type):
        layers = []
        for i in range(len(units)-3):
            layers.append(nn.Linear(units[i],units[i+1]))
            if act_type == 1:
                layers.append(nn.ReLU())
            elif act_type == 2:
                layers.append(nn.Tanh())
            elif act_type == 3:
                layers.append(nn.Sigmoid())
        layers.append(nn.Linear(units[-3],units[-2]))
        return nn.Sequential(*layers)
    
    def _make_layers_decoder(self, units, act_type):
        layers = []
        for i in range(len(units)-2):
            layers.append(nn.Linear(units[-1-i],units[-2-i]))
            if act_type == 1:
                layers.append(nn.ReLU())
            elif act_type == 2:
                layers.append(nn.Tanh())
            elif act_type == 3:
                layers.append(nn.Sigmoid())
        layers.append(nn.Linear(units[1],units[0]))
        return nn.Sequential(*layers)
    
    def reparameterize(self, z_mu, z_logvar):
        eps = torch.randn_like(z_mu)
        z_std = torch.exp(0.5*z_logvar)
        return z_mu + eps * z_std
    
    def encode(self, x):
        x = self.encoder(x)             # [batch_size, hidden_size]
        z_mean = self.z_mean(x)         # [batch_size, latent_size]
        z_logvar = self.z_logvar(x)     # [batch_size, latent_size]
        z_sample = self.reparameterize(z_mean, z_logvar)
        return z_sample, z_mean, z_logvar
    
    def forward(self, x):
        z_sample, z_mean, z_logvar = self.encode(x)
        decoded = self.decoder(z_sample)
        return decoded, z_mean, z_logvar