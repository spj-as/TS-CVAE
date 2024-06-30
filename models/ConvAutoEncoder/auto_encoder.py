import torch
from torch import nn, Tensor
import dgl
from .gat_layer import GATLayer
import logging


# def create_fully_connected_graph(timesteps, batch_size):
#     graphs = []
    
#     for _ in range(batch_size):
#         nodes = list(range(timesteps))
        
#         src = [i for i in nodes for j in nodes if i != j]
#         dst = [j for i in nodes for j in nodes if i != j]
        
#         g = dgl.graph((src, dst))
        
#         graphs.append(g)
    
#     batched_graph = dgl.batch(graphs)
    
#     return batched_graph


# class ConvAutoEncoder(nn.Module):
#     """
#     ConvAutoEncoder is a convolutional autoencoder for a single player
#     """

#     def __init__(self, timesteps: int) -> None:
#         super(ConvAutoEncoder, self).__init__()
#         self.gat = GATLayer(2, 16)
#         # self.conv1d = nn.Conv1d(in_channels=timesteps, out_channels=timesteps, kernel_size=3, padding=1)
#         self.nn = nn.Sequential(
#             nn.Linear(16 , 16//2),
#             nn.Sigmoid(),
#             nn.Linear(16//2, 16//4),
#             nn.Sigmoid(),
#             nn.Linear(16//4, 1),
#         )
#         self.gat_decoder = GATLayer(1, 16)
#         self.decoder = nn.Sequential(
#             nn.Linear(16 , 8),
#             nn.Sigmoid(),
#             nn.Linear(8, 4),
#             nn.Sigmoid(),
#             nn.Linear(4, 2),
#         )

#     @staticmethod
#     def from_checkpoint(path: str, eval: bool = True) -> "ConvAutoEncoder":
#         try:
#             model: torch.nn.Module = ConvAutoEncoder(timesteps=11)
#             weights = torch.load(path)
#             model.load_state_dict(weights)
#             return model.eval() if eval else model
#         except:
#             logging.warning(f"Could not load model from {path}, use default model")
#             return ConvAutoEncoder(timesteps=11)

    # def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
    #     batch_size, timesteps, coords = x.shape
    #     g = create_fully_connected_graph(timesteps, batch_size)
        
        
    #     x_reshaped = x.reshape(batch_size * timesteps, coords)
    #     g = g.to("cuda")
    #     g.ndata['feat'] = x_reshaped
    #     out_gat = self.gat(g, g.ndata['feat'])

    #     out_gat = out_gat.view(batch_size, timesteps, -1)

    #     # out_conv = self.conv1d(out_gat).view(batch_size, timesteps, -1)
    #     out_nn = self.nn(out_gat)

    #     g_decode = create_fully_connected_graph(timesteps, batch_size) 
        
    #     out_nn_reshape = out_nn.view(batch_size * timesteps, 1)
    #     g_decode = g_decode.to("cuda")
    #     g_decode.ndata['feat'] = out_nn_reshape
    #     out_decoded = self.gat_decoder(g_decode, g_decode.ndata['feat'])
    #     out_decoded = self.decoder(out_decoded)
    #     out_decoded = out_decoded.view(batch_size, timesteps, coords)

    #     # out_decoded = self.decoder(out_gat)

    #     return out_nn, out_decoded


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Ensure padding is such that the time dimension remains the same
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class ConvAutoEncoder(nn.Module):
    def __init__(self, input_size, num_channels=[16,32,64], kernel_size=3, dropout=0.2):
        super(ConvAutoEncoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                      dropout=dropout)]

        self.network = nn.Sequential(*layers)

        layers_decoder = []
        num_channels_decode = [64,32,16]
        num_levels = len(num_channels_decode)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = 64 if i == 0 else num_channels_decode[i-1]
            out_channels = num_channels_decode[i]
            layers_decoder += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                      dropout=dropout)]

        self.decode_network = nn.Sequential(*layers_decoder)


        self.nn = nn.Sequential(
            nn.Linear(64 , 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

        self.nn_decode = nn.Sequential(
            nn.Linear(1 , 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16 , 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),
        )

    @staticmethod
    def from_checkpoint(path: str, eval: bool = True) -> "ConvAutoEncoder":
        try:
            model: torch.nn.Module = ConvAutoEncoder(input_size=2)
            weights = torch.load(path)
            model.load_state_dict(weights)
            return model.eval() if eval else model
        except:
            logging.warning(f"Could not load model from {path}, use default model")
            return ConvAutoEncoder(input_size=2)

    def forward(self, x):
        # (batch_size, timesteps, coords)
        # input shape: (batch_size, coords, timesteps)
        x = x.transpose(1, 2)
        out = self.network(x)
        # (batch_size, timesteps, coords)
        out_nn = self.nn(out.transpose(1, 2))
        nn_decode = self.nn_decode(out_nn)
        nn_decode = self.decode_network(nn_decode.transpose(1, 2)).transpose(1, 2)

        out_decoded = self.decoder(nn_decode)

        return out_nn, out_decoded