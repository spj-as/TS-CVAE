import torch
from torch import nn, Tensor
from .tcn_layer import TCNLayer
import logging
from torch_geometric.nn import GATConv
import torch.nn.functional as F

def create_complete_graph_edge_index(num_nodes):
    source_nodes, target_nodes = [], []

    for node in range(num_nodes):
        for target_node in range(num_nodes):
            if node != target_node:
                source_nodes.append(node)
                target_nodes.append(target_node)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    return edge_index



class ShotTypeEmb(nn.Module):
    def __init__(self, locs_feature_dim, shot_feature_dim, gat_output_dim, tcn_output_dim, conv1d_output_dim):
        super(ShotTypeEmb, self).__init__()
        self.gat = GATConv(in_channels=locs_feature_dim, out_channels=gat_output_dim, heads=1)

        self.tcn = TCNLayer(input_size=shot_feature_dim, output_size=tcn_output_dim)

        self.conv1d = nn.Conv1d(in_channels=gat_output_dim , out_channels=conv1d_output_dim, kernel_size=3, padding=1)

        self.shot_nn = nn.Sequential(
            nn.Linear(32,16),
            nn.LeakyReLU(),
            nn.Linear(16,1),
            )
        
        self.locs_nn = nn.Sequential(
            nn.Linear(32,16),
            nn.LeakyReLU(),
            nn.Linear(16,1),
            )
        self.w_locs = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self.w_shot = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        # Reconstruction layers
        self.recon_locs = nn.Linear(1, locs_feature_dim)
        self.recon_shot = nn.Linear(1, shot_feature_dim)

    @staticmethod
    def from_checkpoint(path: str, eval: bool = True) -> "ShotTypeEmb":
        try:
            model: torch.nn.Module = ShotTypeEmb(locs_feature_dim=2, shot_feature_dim=16, gat_output_dim=16, tcn_output_dim=16, conv1d_output_dim=1)
            weights = torch.load(path)
            model.load_state_dict(weights)
            return model.eval() if eval else model
        except:
            logging.warning(f"Could not load ShotTypeEmb model from {path}, use default model")
            return ShotTypeEmb(locs_feature_dim=2, shot_feature_dim=16, gat_output_dim=16, tcn_output_dim=16, conv1d_output_dim=1)

    def forward(self, locs, shot):

        num_nodes = locs.size(1) 
        gat_outs = []
        edge_index = create_complete_graph_edge_index(num_nodes).cuda()
        for i in range(locs.size(0)):
            gat_out = self.gat(locs[i], edge_index)
            gat_out = F.relu(gat_out)
            gat_outs.append(gat_out)

        gat_out = torch.stack(gat_outs, dim=0)  # shape: [batch, nodes, features]

        # TCN
        shot = shot.permute(0, 2, 1)  # [batch, features, time]
        tcn_out = self.tcn(shot)
        
        
        tcn_out = tcn_out.permute(0, 2, 1)  # [batch, time, features]
  
        combined = torch.cat([gat_out, tcn_out], dim=-1)
        # combined = combined.permute(0, 2, 1)  # [batch, features, time]
        # conv1d_out = self.conv1d(combined)
        shot_out = self.shot_nn(combined)
        locs_out = self.locs_nn(combined)
        # conv1d_out = conv1d_out.permute(0, 2, 1)  # [batch, time, features]
        shot_weight = self.w_shot(combined)
        locs_weight = self.w_locs(combined)
        # Reconstruction
        recon_locs = self.recon_locs(locs_out)
        recon_shot = self.recon_shot(shot_out)
        

        return shot_out, locs_out, recon_locs, recon_shot
