import torch
from torch import nn, Tensor
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal,  MultivariateNormal
from argparse import Namespace
from models.ConvAutoEncoder.auto_encoder import ConvAutoEncoder
from models.ShotTypeEmb.auto_encoder import ShotTypeEmb
from models.ConvAutoEncoder.gat_layer import GATLayer, StandardGAT
from models.Dagnet.VAE.encoder import Encoder
from models.Dagnet.VAE.decoder import Decoder
from torch_geometric.nn import GATConv
from models.Dagnet.GAT.gat_model import GAT
from models.Dagnet.GCN.gcn_model import GCN
import numpy as np
import random
import dgl
import random
import matplotlib.pyplot as plt
import seaborn as sns
from utils.adjacency_matrix import (
    compute_vae_adjs_distsim,
    adjs_fully_connected_pred,
    adjs_distance_sim_pred,
    adjs_knn_sim_pred,
)
from utils.eval import (
    mean_square_error,
    average_displacement_error,
    final_displacement_error,
    sample_multinomial,
)
from torch.nn.utils import weight_norm

from .MS_HGNN_batch import MS_HGNN_oridinary, MS_HGNN_hyper, MLP
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Normal:
    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)

    def rsample(self):
        eps = torch.randn_like(self.sigma)
        return self.mu + eps * self.sigma

    def sample(self):
        return self.rsample()

    def kl(self, p=None):
        """compute KL(q||p)"""
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            term1 = (self.mu - p.mu) / (p.sigma + 1e-8)
            term2 = self.sigma / (p.sigma + 1e-8)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return kl

    def mode(self):
        return self.mu


# class RGAT(nn.Module):
#     def __init__(self, hidden_dim):
#         super(RGAT, self).__init__()

#         self.fc_relation1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.fc_relation2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.attention_relation1 = nn.Parameter(torch.Tensor(1, hidden_dim))
#         self.attention_relation2 = nn.Parameter(torch.Tensor(1, hidden_dim))
#         self.leakyrelu = nn.LeakyReLU(0.2)

#         nn.init.xavier_uniform_(self.attention_relation1)
#         nn.init.xavier_uniform_(self.attention_relation2)

#     def forward(self, batch):
#         # batch shape: [batch_size, number_players, hidden_dim]

#         relation1 = batch[:, 0:2, :]  # Players 0 and 1
#         relation2 = batch[:, 2:4, :]  # Players 2 and 3

#         # Compute attention coefficients for each relation
#         attn_coeff_relation1 = F.softmax(
#             self.leakyrelu(
#                 torch.matmul(
#                     self.attention_relation1,
#                     self.fc_relation1(relation1).transpose(1, 2),
#                 )
#             ),
#             dim=-1,
#         )
#         attn_coeff_relation2 = F.softmax(
#             self.leakyrelu(
#                 torch.matmul(
#                     self.attention_relation2,
#                     self.fc_relation2(relation2).transpose(1, 2),
#                 )
#             ),
#             dim=-1,
#         )

#         # Apply attention coefficients to the relations
#         relation1_attended = torch.matmul(attn_coeff_relation1, relation1)

#         relation2_attended = torch.matmul(attn_coeff_relation2, relation2)

#         # Combine attended relations back into one tensor
#         combined = torch.cat(
#             (
#                 relation1_attended,
#                 relation1_attended,
#                 relation2_attended,
#                 relation2_attended,
#             ),
#             dim=1,
#         )
#         return combined


class PastEncoder(nn.Module):
    def __init__(self, args, in_dim=4):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.scale_number = len(args.hyper_scales)

        self.input_fc = nn.Linear(8, self.model_dim)
        self.input_fc2 = nn.Linear(self.model_dim * args.obs_len, self.model_dim)
        self.input_fc3 = nn.Linear(self.model_dim + 2, self.model_dim)

        # self.RGAT = RGAT(hidden_dim=self.model_dim * args.obs_len)

        self.tcn = TemporalConvNet(8, [32, 16,self.model_dim])

        self.gat = GATLayer(
            self.model_dim * args.obs_len, self.model_dim * args.obs_len, 
        )
        self.relationGAT =  StandardGAT(
            8, 16, 8, 
        )

    

    def create_fully_connected_graph(self, timesteps, batch_size):
        graphs = []

        for _ in range(batch_size):
            nodes = list(range(timesteps))

            src = [i for i in nodes for j in nodes if i != j]
            dst = [j for i in nodes for j in nodes if i != j]

            g = dgl.graph((src, dst))

            graphs.append(g)

        batched_graph = dgl.batch(graphs)

        return batched_graph
    
    def construct_team_graph(self, batch_size, num_agents, device, num_timestamps=3):
        g = dgl.batch([dgl.graph(([], [])) for _ in range(batch_size)])

        # Iterate over each graph in the batch
        for i in range(batch_size):
            src, dst = [], []

            # Iterate over each timestamp
            for t1 in range(num_timestamps):
                for j1 in range(num_agents):
                    current_node_index = i * (num_timestamps * num_agents) + t1 * num_agents + j1

                    # Connect to every other node, including itself at other timestamps
                    for t2 in range(num_timestamps):
                        for j2 in range(num_agents):
                            target_node_index = i * (num_timestamps * num_agents) + t2 * num_agents + j2
                            src.append(current_node_index)
                            dst.append(target_node_index)
                # Convert DGL graph to NetworkX graph
            g.add_edges(src, dst)
            

        return g.to(device)

    def forward(self, inputs, batch_size, agent_num):
        length = inputs.shape[1]
        
        gat_in = inputs.reshape(-1, agent_num, inputs.shape[1], inputs.shape[2])
        team1_feat = gat_in[:, :2, :, :]  
        team2_feat = gat_in[:, 2:, :, :] 
        combined_features = []
        for t in range(0,self.args.obs_len-1):
            # Consider only the first 3 timesteps: (batch_size, 2*num_agents*3, feat_dim)
            if t == self.args.obs_len-2:
                team1_feat_in = torch.concat((team1_feat[:, :, t:t+2, :], torch.zeros((team1_feat.shape[0], 2, 1,team1_feat.shape[3]), device=self.args.device)), dim=2)
                team2_feat_in = torch.concat((team2_feat[:, :, t:t+2, :], torch.zeros((team2_feat.shape[0], 2, 1,team2_feat.shape[3]), device=self.args.device)), dim=2)
            else:
                team1_feat_in = team1_feat[:, :, t:t+3, :].reshape(batch_size, -1, inputs.shape[2])
                team2_feat_in = team2_feat[:, :, t:t+3, :].reshape(batch_size, -1, inputs.shape[2])
            team1_graph = self.construct_team_graph(batch_size, 2, self.args.device)
            team2_graph = self.construct_team_graph(batch_size, 2, self.args.device)
            team1_graph.ndata['feat'] = team1_feat_in.reshape(-1, inputs.shape[2])
            team2_graph.ndata['feat'] = team2_feat_in.reshape(-1, inputs.shape[2])
            
            f_gat_team1, _, _ = self.relationGAT(team1_graph, team1_graph.ndata['feat'])
            f_gat_team1 = f_gat_team1.reshape(-1, agent_num //2 , 3, f_gat_team1.shape[1]) 
            
            f_gat_team2, _, _ = self.relationGAT(team2_graph, team2_graph.ndata['feat'])
            f_gat_team2 = f_gat_team2.reshape(-1, agent_num //2 , 3, f_gat_team2.shape[1])
            combined_features.append(torch.cat([f_gat_team1, f_gat_team2], dim=1))
        combined_features = torch.stack(combined_features)
        combined_features = combined_features.reshape(self.args.obs_len-1, -1, agent_num, 3, combined_features.shape[-1])

        combined_features = combined_features.permute(1, 2, 0, 3, 4)
        combined_features = combined_features.reshape(-1, combined_features.shape[2]*combined_features.shape[3]*combined_features.shape[4])
        # relationalGAT_out = self.RGAT(ftraj_input)  # (batch, agents, feat.)
        
        ######0522
        # interaction_feature = []
        # for t in range(self.args.obs_len-1):
        #     # Consider only the first 3 timesteps: (batch_size, 2*num_agents*3, feat_dim)
        #     if t == self.args.obs_len-2:
        #         inter_feat_in = torch.concat((gat_in[:, :, t:t+2, :], torch.zeros((gat_in.shape[0], 4, 1,gat_in.shape[3]), device=self.args.device)), dim=2)
        #     else:
        #         inter_feat_in = gat_in[:, :, t:t+3, :].reshape(batch_size, -1, inputs.shape[2])
        #     inter_graph = self.construct_team_graph(batch_size, 4, self.args.device)
        #     inter_graph.ndata['feat'] = inter_feat_in.reshape(-1, inputs.shape[2])

        #     f_gat_inter = self.relationGAT(inter_graph, inter_graph.ndata['feat'])
        #     f_gat_inter = f_gat_inter.reshape(-1, agent_num  , 3, f_gat_inter.shape[1])
            
        #     interaction_feature.append(f_gat_inter)
        # interaction_feature = torch.stack(interaction_feature)
        # interaction_feature = interaction_feature.reshape(self.args.obs_len-1, -1, agent_num, 3, interaction_feature.shape[-1])
        # interaction_feature = interaction_feature.permute(1, 2, 0, 3, 4)
        # interaction_feature = interaction_feature.reshape(-1, interaction_feature.shape[2]*interaction_feature.shape[3]*interaction_feature.shape[4])

        ######

        tcn_out = (
            inputs.permute(0, 2, 1)
        )
        tcn_out = self.tcn(tcn_out).permute(0, 2, 1)
        tf_in = self.input_fc(tcn_out)  # (batch*agents, time, feat.)

        ftraj_input = tf_in.reshape(
            batch_size, agent_num, length, self.model_dim
        ).reshape(
            batch_size, agent_num, -1
        )  # (batch, agents, time * feat.)
        g = self.create_fully_connected_graph(agent_num, batch_size)
        g = g.to(self.args.device)
        g.ndata["feat"] = ftraj_input.view(batch_size * agent_num, -1)

        gat_out = self.gat(g, g.ndata["feat"]).view(
            batch_size, agent_num, -1
        )  # (batch, agents, feat.)
        gat_out =gat_out.reshape(
            batch_size * agent_num, -1
        )
        
        final_feature = torch.cat(
            (gat_out, combined_features), dim=-1
        )  # (batch, agents, feat.)
        final_feature = gat_out
        # final_feature = torch.cat(
        #     (gat_out, combined_features), dim=-1
        # )  # (batch, agents, feat.)
        # final_feature = combined_features
        output_feature = final_feature.reshape(
            batch_size * agent_num, -1
        )  # (batch * agents, feat.)
        return output_feature


class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation="tanh"):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        initialize_weights(self.affine_layers.modules())

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


class FutureEncoder(nn.Module):
    def __init__(self, args, in_dim=4):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim

        self.input_fc = nn.Linear(8, self.model_dim)
        scale_num = 2 + len(self.args.hyper_scales)
        self.input_fc2 = nn.Linear(self.model_dim * args.pred_len, self.model_dim)
        self.input_fc3 = nn.Linear(self.model_dim + 2, self.model_dim)

        self.out_mlp = MLP2(self.model_dim * 2, [128], "relu")
        self.qz_layer = nn.Linear(
            16 , 2 * self.args.zdim
        )
        initialize_weights(self.qz_layer.modules())

        # self.RGAT = RGAT(hidden_dim=self.model_dim * args.pred_len)
        self.tcn = TemporalConvNet(8, [32,16, self.model_dim])
        self.gat = GATLayer(
            self.model_dim * (args.pred_len + args.obs_len), self.model_dim * args.pred_len)
        self.relationGAT =  StandardGAT(
            8, 16, 8, 
        )


    def create_fully_connected_graph(self, timesteps, batch_size):
        graphs = []

        for _ in range(batch_size):
            nodes = list(range(timesteps))

            src = [i for i in nodes for j in nodes if i != j]
            dst = [j for i in nodes for j in nodes if i != j]

            g = dgl.graph((src, dst))

            graphs.append(g)

        batched_graph = dgl.batch(graphs)

        return batched_graph
    
    def plot_attention_map(self, attn_map, layer_name, time, num_nodes=6):
        num_heads = attn_map.shape[1]
        total_nodes = attn_map.shape[0] // (num_nodes * num_nodes)

        for iter in range(total_nodes):
            fig, axes = plt.subplots(1, num_heads, figsize=(20, 5), sharey=True)
            if num_heads == 1:
                axes = [axes]  # Ensure axes is iterable if there is only one head

            # Extract the attention values for the current chunk and reshape it to [6, 6, num_heads]
            chunk = attn_map[iter*36:(iter+1)*36].view(num_nodes, num_nodes, num_heads)
            
            for i, ax in enumerate(axes):
                # Extract the attention values for the current head
                attn_values = chunk[:, :, i].cpu().detach().numpy()
                
                sns.heatmap(attn_values, cmap='viridis', ax=ax, cbar=True, annot=True, fmt=".2f")
                ax.set_title(f'Head {i+1}')
                ax.set_xlabel('Nodes')
                if i == 0:
                    ax.set_ylabel('Nodes')
            fig.suptitle(f'Attention Map for {layer_name} at {str(time)}, Chunk {iter + 1}')
            plt.savefig(f'./atten/Attention_Map_for_{layer_name}_{str(time)}_Chunk_{iter + 1}.png')
            plt.show()

    def construct_team_graph(self, batch_size, num_agents, device, num_timestamps=3):
        g = dgl.batch([dgl.graph(([], [])) for _ in range(batch_size)])
        for i in range(batch_size):
            src, dst = [], []
            for t in range(num_timestamps):  # num_timestamps timesteps
                for j in range(num_agents):  # num_agents here should be 2 (one team)
                    current_node_index = i * (num_timestamps * num_agents) + t * num_agents + j
                    # Connect to other agent in the same team at the same timestep
                    for k in range(num_agents):
                        if j != k:
                            src.append(current_node_index)
                            dst.append(i * (num_timestamps * num_agents) + t * num_agents + k)
                    # Connect to itself at the next timestep
                    if t < 2:
                        src.append(current_node_index)
                        dst.append(current_node_index + num_agents)
            g.add_edges(src, dst)
        return g.to(device)
    def forward(self, inputs, batch_size, agent_num, past_feature):
        length = inputs.shape[1]
        
        # relationalGAT_out = self.RGAT(ftraj_input)
        gat_in = inputs.reshape(-1, agent_num, inputs.shape[1], inputs.shape[2])
        team1_feat = gat_in[:, :2, :, :]  
        team2_feat = gat_in[:, 2:, :, :] 
        combined_features = []
        # for t in range(0,self.args.obs_len + self.args.pred_len -1): 
        for t in range(0, self.args.obs_len + self.args.pred_len -1): 
            # Consider only the first 3 timesteps: (batch_size, 2*num_agents*3, feat_dim)
            if t == self.args.obs_len +self.args.pred_len-2:
                team1_feat_in = torch.concat((team1_feat[:, :, t:t+2, :], torch.zeros((team1_feat.shape[0], 2, 1,team1_feat.shape[3]), device=self.args.device)), dim=2)
                team2_feat_in = torch.concat((team2_feat[:, :, t:t+2, :], torch.zeros((team2_feat.shape[0], 2, 1,team2_feat.shape[3]), device=self.args.device)), dim=2)
            else:
                team1_feat_in = team1_feat[:, :, t:t+3, :].reshape(batch_size, -1, inputs.shape[2])
                team2_feat_in = team2_feat[:, :, t:t+3, :].reshape(batch_size, -1, inputs.shape[2])
            team1_graph = self.construct_team_graph(batch_size, 2, self.args.device)
            team2_graph = self.construct_team_graph(batch_size, 2, self.args.device)
        
            team1_graph.ndata['feat'] = team1_feat_in.reshape(-1, inputs.shape[2])
            team2_graph.ndata['feat'] = team2_feat_in.reshape(-1, inputs.shape[2])

            f_gat_team1, att1, att2 = self.relationGAT(team1_graph, team1_graph.ndata['feat'])
            
            # self.plot_attention_map(att2, 'Layer 2', t)
            f_gat_team1 = f_gat_team1.reshape(-1, agent_num //2 , 3, f_gat_team1.shape[1])
            f_gat_team2, att1, att2 = self.relationGAT(team2_graph, team2_graph.ndata['feat'])
            f_gat_team2 = f_gat_team2.reshape(-1, agent_num //2 , 3, f_gat_team2.shape[1])
            combined_features.append(torch.cat([f_gat_team1, f_gat_team2], dim=1))
        combined_features = torch.stack(combined_features)
        combined_features = combined_features.reshape(self.args.obs_len +self.args.pred_len-1, -1, agent_num, 3, combined_features.shape[-1])

        # combined_features = combined_features.reshape(self.args.obs_len +self.args.pred_len-1, -1, agent_num, 5, combined_features.shape[-1])
        combined_features = combined_features.permute(1, 2, 0, 3, 4)
        combined_features = combined_features.reshape(-1, combined_features.shape[2]*combined_features.shape[3]*combined_features.shape[4])
        

        ######0522
        # interaction_feature = []
        # for t in range(self.args.obs_len-1):
        #     # Consider only the first 3 timesteps: (batch_size, 2*num_agents*3, feat_dim)
        #     if t == self.args.obs_len-2:
        #         inter_feat_in = torch.concat((gat_in[:, :, t:t+2, :], torch.zeros((gat_in.shape[0], 4, 1,gat_in.shape[3]), device=self.args.device)), dim=2)
        #     else:
        #         inter_feat_in = gat_in[:, :, t:t+3, :].reshape(batch_size, -1, inputs.shape[2])
        #     inter_graph = self.construct_team_graph(batch_size, 4, self.args.device)
        #     inter_graph.ndata['feat'] = inter_feat_in.reshape(-1, inputs.shape[2])

        #     f_gat_inter = self.relationGAT(inter_graph, inter_graph.ndata['feat'])
        #     f_gat_inter = f_gat_inter.reshape(-1, agent_num , 3, f_gat_inter.shape[1])
            
        #     interaction_feature.append(f_gat_inter)
        # interaction_feature = torch.stack(interaction_feature)
        # interaction_feature = interaction_feature.reshape(self.args.obs_len-1, -1, agent_num, 3, interaction_feature.shape[-1])
        # interaction_feature = interaction_feature.permute(1, 2, 0, 3, 4)
        # interaction_feature = interaction_feature.reshape(-1, interaction_feature.shape[2]*interaction_feature.shape[3]*interaction_feature.shape[4])

            ######
        tcn_out = (
            inputs
            .view(batch_size * agent_num, length, -1)
            .permute(0, 2, 1)
        )
        tcn_out = self.tcn(tcn_out).permute(0, 2, 1)

        tf_in = self.input_fc(tcn_out)

        ftraj_input = tf_in.reshape(
            batch_size, agent_num, length, self.model_dim
        ).reshape(batch_size, agent_num, -1)

        g = self.create_fully_connected_graph(agent_num, batch_size)
        g = g.to(self.args.device)
        g.ndata["feat"] = ftraj_input.view(batch_size * agent_num, -1)

        gat_out = self.gat(g, g.ndata["feat"]).view(batch_size, agent_num, -1)
        # final_feature = torch.cat((gat_out, relationalGAT_out), dim=-1)
        gat_out =gat_out.reshape(
            batch_size * agent_num, -1
        )
        # final_feature = combined_features
        # final_feature = torch.cat(
        #     (gat_out, combined_features), dim=-1
        # )  # (batch, agents, feat.)

        final_feature = torch.cat(
            (gat_out, combined_features), dim=-1
        )  # (batch, agents, feat.)
        final_feature = gat_out


        # final_feature = torch.cat((final_feature, past_feature), dim=-1)
        final_feature = final_feature.reshape(batch_size * agent_num, -1)
        q_z_params = self.qz_layer(final_feature)

        return q_z_params


class DecomposeBlock(nn.Module):
    """
    Balance between reconstruction task and prediction task.
    """

    def __init__(self, past_len, future_len, input_dim):
        super(DecomposeBlock, self).__init__()
        # * HYPER PARAMETERS
        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        dim_embedding_key = 96
        self.past_len = past_len
        self.future_len = future_len

        self.conv_past = nn.Conv1d(
            channel_in, channel_out, dim_kernel, stride=1, padding=1
        )
        # self.encoder_past = nn.GRU(channel_out, dim_embedding_key, 1, batch_first=True)
        self.encoder_past = nn.ModuleList(
            [
                nn.GRU(channel_out, dim_embedding_key, 1, batch_first=True)
                for i in range(future_len)
            ]
        )
        

        self.decoder_y = MLP(
                input_dim*5+4,
                4,
                hidden_size=(256, 128),
            )
        
        self.decoder_shot = nn.Sequential(
            MLP(
                input_dim*5+16+4+4,
                16,
                hidden_size=(256, 128),
            ), 
            nn.Softmax(),
        )
        self.decoder_hit = nn.Sequential(
            MLP(
                input_dim*5+4+4,
                4,
                hidden_size=(256, 128),
            ), 
            nn.Softmax(),
        )
           
        

        self.decoder_x = MLP(
            input_dim * future_len +dim_embedding_key , past_len * 2, hidden_size=(512, 256)
        )

        self.coors_weights = nn.Sequential(
            nn.Linear(input_dim * future_len, input_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.LeakyReLU(),
        )
   
  
        self.relu = nn.ReLU()

        self.gat = GATLayer(input_dim+4, input_dim*2)

        self.tcn = TemporalConvNet(
            input_dim +4,
            [16, 16, 32],
        )

        # self.nn = nn.Sequential(
        #     nn.Linear(input_dim * future_len , input_dim * future_len),
        #     nn.Dropout(0.7),
        #     nn.Linear(input_dim * future_len,  input_dim * future_len),
        #     nn.Dropout(0.7),
        # )

        # kaiming initialization
        self.mu = nn.Sequential(
            nn.Linear(48, 2),
            
        )
        self.sigma = nn.Sequential(
            nn.Linear(48, 2),
            nn.Sigmoid(),
        )
        self.rho = nn.Sequential(
            nn.Linear(48, 1),
            nn.Tanh()
        )
        self.nn = nn.Sequential(
            nn.Linear(48, 96),
            nn.LeakyReLU(),
            nn.Linear(96, 48)
        )
        self.conv = nn.Conv1d(
            input_dim +4, input_dim*2, dim_kernel, stride=1, padding=1
        )
      

        self.init_parameters()
    def sample_bivariate_normal(self, mu_x, mu_y, sigma_x, sigma_y, rho, n_samples):   
        
        mean = torch.stack([mu_x, mu_y], dim=-1)
        
        cov_xx = sigma_x.pow(2)
        cov_yy = sigma_y.pow(2)
        cov_xy = rho * sigma_x * sigma_y
        
        epsilon = 1e-6 
        cov_matrices = torch.stack([
            torch.stack([cov_xx + epsilon, cov_xy], dim=-1),
            torch.stack([cov_xy, cov_yy + epsilon], dim=-1)
        ], dim=-1)
        
        distribution = MultivariateNormal(mean, covariance_matrix=cov_matrices)
        samples = distribution.sample()
        
        return samples

    
    def create_fully_connected_dgl_graph(self, num_nodes):
        """
        Create a DGLGraph representing a fully connected graph with num_nodes nodes.
        """
        src, dst = zip(
            *[(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        )
        g = dgl.graph((src, dst))
        return g

    def batch_fully_connected_graphs(self, batch_size, num_nodes):
        """
        Create a batch of fully connected graphs and return as a single DGLGraph.
        """
        graphs = [
            self.create_fully_connected_dgl_graph(num_nodes) for _ in range(batch_size)
        ]
        batched_graph = dgl.batch(graphs)
        return batched_graph

    def init_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        # nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        # nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)

        nn.init.zeros_(self.conv_past.bias)
        # nn.init.zeros_(self.encoder_past.bias_ih_l0)
        # nn.init.zeros_(self.encoder_past.bias_hh_l0)

    def forward(
        self,
        # x_true,
        # x_hat,
        f,
        num_agents,
        num_samples,
        cur_locs,
        cur_shot,
        cur_hit,
        forcast,
        tf_threshold,
    ):
        """
        >>> Input:
            x_true: N, T_p, 2
            x_hat: N, T_p, 2
            f: N, D

        >>> Output:
            x_hat_after: N, T_p, 2
            y_hat: n, T_f, 2
        """
       
        # x_ = x_true - x_hat
        # x_true = torch.transpose(x_true, 1, 2)
        # past_embed = self.relu(self.conv_past(x_true))
        # past_embed = torch.transpose(past_embed, 1, 2)

        # _, state_past = self.encoder_past(past_embed)

        # state_past = state_past.squeeze(0)
        y_hat_list = []
        s_hat_list=[]
        p_hat_list=[]
     
        if (forcast):
            cur_locs = cur_locs.repeat_interleave(num_samples, dim=0)
            cur_shot = cur_shot.repeat_interleave(num_samples, dim=0)
            cur_hit = cur_hit.repeat_interleave(num_samples, dim=0)

            for step in range(self.future_len):
                input_feat = torch.concat((f,cur_locs), -1)
                Ginput_feat = input_feat.reshape(-1, num_agents, input_feat.shape[-1])
                g = self.batch_fully_connected_graphs(Ginput_feat.shape[0], num_agents).to(f.device)
                f_flat = Ginput_feat.view(-1, Ginput_feat.shape[-1])
                g.ndata["feat"] = f_flat
                f_gat = self.gat(g, g.ndata["feat"])
                Tinput_feat = input_feat.reshape(-1, num_agents, input_feat.shape[-1]).permute(0,2,1)
                f_cov = self.tcn(Tinput_feat).reshape(input_feat.shape[0], -1)
                
                decode_feat = torch.concat((f_flat, f_gat, f_cov), -1)


                y_hat = self.decoder_y(decode_feat)
                p_hat = self.decoder_hit(torch.concat((decode_feat, cur_hit), -1))
                s_hat = self.decoder_shot(torch.concat((decode_feat, cur_shot, cur_hit), -1))
                
                cur_locs = cur_locs.squeeze(1)
                y_hat += cur_locs
                
                y_hat_list.append(y_hat)
                s_hat_list.append(s_hat)
                p_hat_list.append(p_hat)

                # means.append(mu)
                # variances.append(sigma)
                # rhos.append(rho)
                cur_locs = y_hat
                cur_shot = s_hat
                cur_hit = p_hat
            
        else:
            cur_locs_list = cur_locs
            cur_shot_list = cur_shot
            cur_hit_list  =cur_hit
            for step in range(self.past_len + self.future_len -1):
                input_feat = torch.concat((f,cur_locs_list[:,step]), -1)
                if step > 0 and tf_threshold < random.random():
                    input_feat = torch.concat((f,y_hat_list[-1]), -1)
                
                Ginput_feat = input_feat.reshape(-1, num_agents, input_feat.shape[-1])
                g = self.batch_fully_connected_graphs(Ginput_feat.shape[0], num_agents).to(f.device)
                f_flat = Ginput_feat.view(-1, Ginput_feat.shape[-1])
                g.ndata["feat"] = f_flat
                f_gat = self.gat(g, g.ndata["feat"])
                Tinput_feat = input_feat.reshape(-1, num_agents, input_feat.shape[-1]).permute(0,2,1)
                f_cov = self.tcn(Tinput_feat).reshape(input_feat.shape[0], -1)
                decode_feat = torch.concat((f_flat, f_gat, f_cov), -1)
              

                
                y_hat = self.decoder_y(decode_feat)
                cur_locs = cur_locs_list[:,step].squeeze(1)
                cur_shot = cur_shot_list[:,step]
                cur_hit = cur_hit_list[:,step]
                cur_hit_in = cur_hit
                cur_shot_in = cur_shot
                if step > 0 and tf_threshold < random.random():
                    cur_hit_in = p_hat_list[-1]
                if step > 0 and tf_threshold < random.random():
                    cur_shot_in = s_hat_list[-1]
                p_hat = self.decoder_hit(torch.concat((decode_feat,cur_hit_in), -1))
                s_hat = self.decoder_shot(torch.concat((decode_feat, cur_shot_in, cur_hit_in), -1))
                
                y_hat += cur_locs
                cur_locs = y_hat
                        
                y_hat_list.append(y_hat)
                s_hat_list.append(s_hat)
                p_hat_list.append(p_hat)
                # means.append(mu)
                # variances.append(sigma)
                # rhos.append(rho)
        y_hat_list = torch.stack(y_hat_list).permute(1,0,2)[:,:,:2]
        s_hat_list = torch.stack(s_hat_list).permute(1,0,2)
        p_hat_list = torch.stack(p_hat_list).permute(1,0,2)
        # means = torch.stack(means).permute(1,0,2)
        # variances = torch.stack(variances).permute(1,0,2) 
        # rhos = torch.stack(rhos).permute(1,0,2)    
        

        # f = self.tcn(f.reshape(f.shape[0]//num_agents, num_agents, -1).permute(0,2,1)).permute(0,2,1)
        # f = f.reshape(-1,f.shape[-1])

        # coors_input = self.coors_weights(f)
        # shot_input = self.shots_weights(f)
        # hit_input = self.hits_weights(f)  
       
        
        # drop_feat = self.nn(input_feat)
        
            # cur_locs = cur_locs.unsqueeze(1)
            # if step != 0:
            #     x_true = torch.concat((x_true, cur_locs.permute(0,2,1)), dim=-1).permute(0,2,1)
            # x_true = torch.transpose(x_true, 1, 2)
            # past_embed = self.relu(self.conv_past(x_true))
            # past_embed = torch.transpose(past_embed, 1, 2)
            # _, state_past = self.encoder_past[step](past_embed)

            # state_past = state_past.squeeze(0)
            # input_feat = torch.cat((f, state_past), dim=1)
            
            # input_ = torch.cat((input_feat, cur_locs), dim=-1)
            
        # x_hat_after = self.decoder_x(input_feat).contiguous().view(-1, self.past_len, 2)

        return  y_hat_list, s_hat_list,p_hat_list


class TrajDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.decode_way = "RES"
        scale_num = 2 + len(self.args.hyper_scales)

        self.num_decompose = args.num_decompose
        self.past_length = self.args.obs_len
        self.future_length = self.args.pred_len
        input_dim = self.args.zdim


        self.decompose = nn.ModuleList(
            [
                DecomposeBlock(self.past_length, self.future_length, input_dim)
                for _ in range(self.num_decompose)
            ]
        )
        # self.decompose = nn.ModuleList([DecomposeBlock(self.past_length, self.future_length, input_dim) for _ in range(self.num_decompose)])

    def forward(
        self,
        past_feature,
        z,
        batch_size_curr,
        agent_num_perscene,
        past_traj,
        cur_location,
        cur_shot, 
        cur_hit,
        sample_num,
        mode="train",
        forcast= False,
        tf_threshold =1,
    ):
        agent_num = batch_size_curr * agent_num_perscene

        # past_traj_repeat = past_traj.repeat_interleave(sample_num, dim=0)
     
        # if sample_num == 1:
        #     past_traj_repeat = past_traj_repeat.unsqueeze(0)

        # past_feature = past_traj_repeat.permute(1, 0, 2, 3)
        # past_feature = past_feature.view(past_feature.shape[0], sample_num, -1)

        z_in = z.view(-1, sample_num, z.shape[-1])

        hidden = z_in
        
        hidden = hidden.view(agent_num * sample_num, -1)
        # x_true = past_traj_repeat.reshape(
        #     -1, self.past_length, 2
        # ).clone()  # torch.transpose(pre_motion_scene_norm, 0, 1)
        # x_hat = torch.zeros_like(x_true)
        # batch_size = x_true.size(0)
        batch_size = z_in.size(0)
        prediction = torch.zeros((batch_size* sample_num, self.future_length, 2)).cuda()
        prediction_shot = torch.zeros((batch_size* sample_num, self.future_length, 16)).cuda()
        prediction_hit = torch.zeros((batch_size* sample_num, self.future_length, 4)).cuda()
      
        if not forcast:
            prediction = torch.zeros((batch_size, self.past_length + self.future_length-1, 2)).cuda()
            prediction_shot = torch.zeros((batch_size* sample_num, self.past_length + self.future_length-1, 16)).cuda()
            prediction_hit = torch.zeros((batch_size* sample_num, self.past_length + self.future_length-1, 4)).cuda()
      
        for i in range(self.num_decompose):
            y_hat, s_hat, p_hat= self.decompose[i](
                # x_true,
                # x_hat,
                hidden,
                self.args.num_agents,
                sample_num,
                cur_location,
                cur_shot,
                cur_hit,
                forcast,
                tf_threshold,
            )
            prediction += y_hat
            prediction_shot += s_hat
            prediction_hit += p_hat
      
            # reconstruction += x_hat
        norm_seq = prediction.view(agent_num * sample_num, -1, 2)
        norm_shot = prediction_shot.view(agent_num * sample_num, -1, 16)
        norm_hit = prediction_hit.view(agent_num * sample_num, -1, 4)

  
       
        
        # recover_pre_seq = reconstruction.view(
        #     agent_num * sample_num, self.past_length, 2
        # )

        # norm_seq = norm_seq.permute(2,0,1,3).view(self.future_length, agent_num * sample_num,2)

        # cur_location_repeat = cur_location.repeat_interleave(sample_num, dim=0)
        # out_seq = (
        #     norm_seq + cur_location_repeat
        # )  # (agent_num*sample_num,self.past_length,2)
        
        out_seq = norm_seq
        
        if mode == "inference":
            out_seq = out_seq.view(
                -1, sample_num, *out_seq.shape[1:]
            )  # (agent_num,sample_num,self.past_length,2)
            norm_shot = norm_shot.view(
                -1, sample_num, *norm_shot.shape[1:]
            ) 
            norm_hit = norm_hit.view(
                -1, sample_num, *norm_hit.shape[1:]
            ) 
            # mean = mean.view(
            #     -1, sample_num, *mean.shape[1:]
            # )
            # var = var.view(
            #     -1, sample_num, *var.shape[1:]
            # )
            # rho = rho.view(
            #     -1, sample_num, *var.shape[1:]
            # )
        return out_seq, norm_shot, norm_hit


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return F.relu(out)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DAGNet(nn.Module):
    def __init__(self, args: Namespace, n_max_player: int):
        super(DAGNet, self).__init__()

        self.n_max_player: int = n_max_player
        self.n_layers: int = args.n_layers
        self.x_dim: int = args.x_dim
        self.h_dim: int = args.hidden_dim
        self.z_dim: int = args.z_dim
        self.p_dim: int = args.p_dim
        self.d_dim: int = n_max_player * 2
        self.s_dim: int = args.s_dim
        self.rnn_dim: int = args.rnn_dim
        self.batch_size: int = args.batch_size
        self.obs_len: int = args.obs_len
        self.pred_len: int = args.pred_len

        # self.graph_model: str = args.graph_model
        self.graph_hid: int = args.graph_hid
        self.adjacency_type: str = args.adjacency_type
        self.top_k_neigh: int = args.top_k_neigh
        self.sigma = args.sigma
        self.alpha: float = args.alpha
        self.n_heads: int = args.n_heads
        self.device: str = args.device

        # Embeddings
        self.position_emb = ConvAutoEncoder.from_checkpoint(
            args.position_emb_path, eval=True
        )
        self.shot_type_emb = ShotTypeEmb.from_checkpoint(args.shot_emb_path, eval=True)
        self.prior_emb = nn.Conv1d(
            in_channels=self.obs_len, out_channels=16, kernel_size=3, padding=1
        )
        self.encode_emb = nn.Conv1d(
            in_channels=self.obs_len + self.pred_len,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        # RNN
        self.tcn = TemporalConvNet(self.x_dim + 1, [32, self.rnn_dim])

        # Encoders
        self.past_encoder = PastEncoder(args)
        self.all_encoder = FutureEncoder(args)
        scale_num = 2 + len(args.hyper_scales)
        # args.z_dim//2 * args.obs_len  +
        # self.pz_layer = nn.Linear( (args.obs_len-1)*24 *4 *2 , 2 * args.z_dim)
        self.pz_layer = nn.Linear(80, 2 * args.z_dim)
        self.q_layer = nn.Linear(2 * args.zdim, 2 * args.z_dim)

        # Decoders
        self.decoder = TrajDecoder(args)

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def _nll_gauss(self, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        x1 = torch.sum(((x - mu).pow(2)) / torch.exp(logvar), dim=1)
        x2 = x.size(1) * np.log(2 * np.pi)
        x3 = torch.sum(logvar, dim=1)
        nll = torch.mean(0.5 * (x1 + x2 + x3))
        return nll

    def _kld(self, mean_enc, logvar_enc, mean_prior, logvar_prior):
        x1 = torch.sum((logvar_prior - logvar_enc), dim=1)
        x2 = torch.sum(torch.exp(logvar_enc - logvar_prior), dim=1)
        x3 = torch.sum(
            (mean_enc - mean_prior).pow(2) / (torch.exp(logvar_prior)), dim=1
        )
        kld_element = x1 - mean_enc.size(1) + x2 + x3
        return torch.mean(0.5 * kld_element)

    def calculate_loss_pred(self, pred, target, batch_size):
        loss = (target - pred).pow(2).sum()
        # max_values, _ = torch.max(loss, dim=1)
        # loss = max_values.sum()
        loss /= batch_size
        loss /= pred.shape[1]
        return loss

    def calculate_loss_kl(
        self, qz_distribution, pz_distribution, batch_size, agent_num, min_clip
    ):
        loss = qz_distribution.kl(pz_distribution).sum()
        loss /= (batch_size * agent_num)
        loss_clamp = loss.clamp_min_(min_clip)
        return loss_clamp

    def calculate_loss_recover(self, pred, target, batch_size):
        loss = (target - pred).pow(2).sum()
        loss /= batch_size
        loss /= pred.shape[1]
        return loss

    def calculate_loss_diverse(self, pred, target, batch_size):
        diff = target.unsqueeze(1) - pred
        avg_dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
        loss = avg_dist.min(dim=1)[0]
        loss = loss.mean()
        return loss
    
    
    def Gaussian2D_batch_loss(self, V_mu, V_sigma, V_trgt, rho):
        rho = rho.squeeze(-1)
        
        normx = V_trgt[:, :, 0] - V_mu[:, :, 0]
        normy = V_trgt[:, :, 1] - V_mu[:, :, 1]

        sx = torch.exp(V_sigma[:, :, 0]) 
        sy = torch.exp(V_sigma[:, :, 1])  
        corr = torch.tanh(rho)  

        sxsy = sx * sy
        z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
        negRho = 1 - corr ** 2

        result = torch.exp(-z / (2 * negRho))
        denom = 2 * np.pi * sxsy * torch.sqrt(negRho)

        result = result / denom

        epsilon = 1e-20

        result = -torch.log(torch.clamp(result, min=epsilon))
        result = torch.sum(result) / (V_trgt.shape[0])

        return result

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()


    def generate_positional_encodings(self, length, depth):
        """
        Generate positional encodings for a sequence.

        Parameters:
        - length: The length of the sequence.
        - depth: The depth of the positional encoding.

        Returns:
        - A numpy array of shape (length, depth) containing the positional encodings.
        """
        assert depth % 2 == 0, "Depth must be an even number."
        
        position = np.arange(length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, depth, 2) * -(np.log(10000.0) / depth))
        positional_encoding = np.zeros((length, depth))
        positional_encoding[:, 0::2] = np.sin(position * div_term)
        positional_encoding[:, 1::2] = np.cos(position * div_term)
        
        return positional_encoding

    def positional_encoding(self, input, depth):
        """
        Apply positional encoding to an input tensor with shape [batch_size, agent_num, length, 2].
        
        Parameters:
        - input_tensor: A numpy array of shape (batch_size, agent_num, length, 2).
        - depth: The depth of the positional encoding to be applied.
        
        Returns:
        - A numpy array with positional encodings applied.
        """
        batch_size, agent_num, length, _ = input.shape
        pe = self.generate_positional_encodings(length, depth)  
        pe_expanded = pe[np.newaxis, np.newaxis, :, :]
        pe_expanded = np.repeat(pe_expanded, batch_size, axis=0)
        pe_expanded = np.repeat(pe_expanded, agent_num, axis=1)
        
        # Concatenate the positional encodings with the input tensor along the last dimension.
        # Assuming you want to concatenate. If you want to add, you need to ensure the last dimension matches and then simply add.
        # input_with_pe = np.concatenate((input.cpu(), pe_expanded), axis=-1)
        input_with_pe = torch.tensor(pe_expanded)       
        return input_with_pe

    def forward(
        self,
        traj: Tensor,
        traj_rel: Tensor,
        group: Tensor,
        shot_type_ohe: Tensor,
        hit_player: Tensor,
        direction:Tensor,
        mask: bool,
        tf_threshold: float,
        obs_len: int,
        args,
        epoch: int,
        beta: float,
    ):

        batch, timesteps, num_agents, features = traj.shape
        traj_with_emb = []
        for i in range(num_agents):
            out, _ = self.position_emb(traj[:, :, i, :])
            traj_with_emb.append(out.squeeze(1))
        traj_with_emb = torch.stack(traj_with_emb)
        
        ##### add positional encoding ######
        pos_input = traj
        # input_with_pe = self.positional_encoding(pos_input, 8).clone().detach().to(args.device)


        add_emb = []
        for i, player_emb in enumerate(traj_with_emb):
            new_coords = torch.concat((pos_input[:, :, i, :], player_emb), dim=-1)
            add_emb.append(new_coords)
        # x_emb = torch.stack(add_emb).permute(0, 2, 1, 3).reshape(-1, timesteps, 2 + 1)
        x_emb = torch.stack(add_emb).permute(1, 0, 2, 3)
        x_emb = x_emb.reshape(-1, x_emb.shape[2], x_emb.shape[3]) # batch *num_agents , time, coor

        #### add shot type & coordinate encoding ######
        
        shot_with_emb = []
        traj_with_emb = []
        for i in range(num_agents):
            shot_out, locs_out, _, _ = self.shot_type_emb(traj[:, :, i, :], shot_type_ohe[: ,: ,i ,: ])
            shot_with_emb.append(shot_out.squeeze(1))
            traj_with_emb.append(locs_out.squeeze(1))
        traj_with_emb = torch.stack(traj_with_emb).permute(1,0,2,3)
        traj_with_emb =traj_with_emb.reshape(-1, traj_with_emb.shape[2], traj_with_emb.shape[3])
        shot_with_emb = torch.stack(shot_with_emb).permute(1,0,2,3)
        shot_with_emb = shot_with_emb.reshape(-1, shot_with_emb.shape[2], shot_with_emb.shape[3])
        
        shot_input = (
            shot_type_ohe.permute(0, 2, 1, 3)
            .reshape(-1, timesteps, args.s_dim)
            .to(self.device)
        )

        hit_input = (
            hit_player.permute(0, 2, 1, 3)
            .reshape(-1, timesteps, args.p_dim)
            .to(self.device)
        )
        
        
        traj_rel = traj_rel.permute(0, 2, 1, 3)
        traj_rel = traj_rel.reshape(-1, traj_rel.shape[2], traj_rel.shape[3])
        # emb_feats = torch.concat((x_emb, traj_rel), dim=-1)
        # feats_pos = self.positional_encoding(emb_feats.reshape(-1,num_agents,self.obs_len+self.pred_len,5), 8).clone().detach().to(args.device)
        # feats_pos = feats_pos.reshape(-1,self.obs_len+self.pred_len, feats_pos.shape[-1])
        emb_feats = torch.concat((x_emb[:,:,:2], traj_with_emb, traj_rel, shot_input, shot_with_emb, hit_input), dim=-1)
        feats_pos = self.positional_encoding(emb_feats.reshape(-1,num_agents,self.obs_len+self.pred_len, 26), 8).clone().detach().to(args.device)
        feats_pos = feats_pos.reshape(-1,self.obs_len+self.pred_len, feats_pos.shape[-1])
        real_traj = traj.permute(0, 2, 1, 3)
        real_traj = real_traj.reshape(-1, real_traj.shape[2],  real_traj.shape[3])


        feats = torch.concat((real_traj, traj_rel), dim=-1)
        past_traj = real_traj[:, : self.obs_len, :]
        future_traj = real_traj[:, self.obs_len :, :]

        ## emb_position + velocity
        past_feats_emb = feats_pos[:, : self.obs_len, :].float() 
        future_feats_emb = feats_pos[:, self.obs_len :, :].float() 
       

        direction_input = (
            direction.reshape(timesteps, -1, args.num_agents, 2)
            .permute(1, 2, 0, 3)
            .reshape(-1, timesteps, 2)
            .to(self.device)
        )

        cur_location = feats
        h = torch.zeros(self.n_layers, batch, self.rnn_dim).to(self.device)

        ########################## CVAE encoder ##########################
        past_direction = direction_input[:, : self.obs_len, :]
        future_direction = direction_input[:, self.obs_len :, :]
        # past_feats_emb = torch.concat((past_feats_emb),dim=-1)
        # future_feats_emb = torch.concat((future_feats_emb),dim=-1)
        past_shot = shot_input[:, : self.obs_len, :]
        future_shot = shot_input[:, self.obs_len :, :]
        cur_shot = shot_input

        past_hit = hit_input[:, : self.obs_len, :]
        future_hit = hit_input[:, self.obs_len :, :]
        cur_hit = hit_input

        past_feats_emb_shot = past_feats_emb
        future_feats_emb_shot = future_feats_emb
        past_feature = self.past_encoder(
            past_feats_emb_shot,
            batch,
            args.num_agents,
        )

        all_feature = self.all_encoder(
            feats_pos.float(),
            batch,
            args.num_agents,
            past_feature
        )
        ########################## CVAE decoder ##########################
        prior_input = past_feature
        encoder_input = all_feature

        ### q dist ###
        if args.ztype == "gaussian":
            q_param = self.q_layer(encoder_input)
            qz_distribution = Normal(params=q_param)
        else:
            ValueError("Unknown hidden distribution!")
        qz_sampled = qz_distribution.rsample()

        ### p dist ###
        if args.learn_prior:
            pz_param = self.pz_layer(prior_input)
            if args.ztype == "gaussian":
                pz_distribution = Normal(params=pz_param)

            else:
                ValueError("Unknown hidden distribution!")
        else:
            if args.ztype == "gaussian":
                pz_distribution = Normal(
                    mu=torch.zeros(prior_input.shape[0], args.zdim).to(traj.device),
                    logvar=torch.zeros(prior_input.shape[0], args.zdim).to(traj.device),
                )
            else:
                ValueError("Unknown hidden distribution!")

        loss_kl = self.calculate_loss_kl(
            qz_distribution, pz_distribution, batch, args.num_agents, min_clip=2.0
        )

        ### p dist for best 20 loss ###
        sample_num = 1
        if args.learn_prior:
            past_feature_repeat = prior_input.repeat_interleave(sample_num, dim=0)
            p_z_params = self.pz_layer(past_feature_repeat)
            if args.ztype == "gaussian":
                pz_distribution = Normal(params=p_z_params)
            else:
                ValueError("Unknown hidden distribution!")
        else:
            past_feature_repeat = prior_input.repeat_interleave(sample_num, dim=0)
            if args.ztype == "gaussian":
                pz_distribution = Normal(
                    mu=torch.zeros(past_feature_repeat.shape[0], args.zdim).to(
                        traj.device
                    ),
                    logvar=torch.zeros(past_feature_repeat.shape[0], args.zdim).to(
                        traj.device
                    ),
                )
            else:
                ValueError("Unknown hidden distribution!")

        pz_sampled = pz_distribution.rsample()

        
        diverse_pred_traj, pred_shot, pred_hit= self.decoder(
            past_feature_repeat,
            qz_sampled,
            batch,
            args.num_agents,
            past_traj,
            cur_location,
            cur_shot,
            cur_hit,
            sample_num=sample_num,
            tf_threshold =tf_threshold
        )
        # loss_recover = self.calculate_loss_recover(recover_traj, past_traj, batch)
        
        loss_pred = self.calculate_loss_pred(diverse_pred_traj, real_traj[:,1:], batch)
        # nll = self.Gaussian2D_batch_loss(mean, var, real_traj[:,1:], rho)
        loss_ade = average_displacement_error(
            pred=diverse_pred_traj, actual=real_traj, pred_len=args.pred_len+args.obs_len-1,skipFirst=True
        )
        lose_fde = final_displacement_error(
            pred=diverse_pred_traj, actual=real_traj, pred_len=args.pred_len+args.obs_len-1,skipFirst=True
        )
        lose_mse = mean_square_error(
            pred=diverse_pred_traj, actual=real_traj, pred_len=args.pred_len+args.obs_len-1,skipFirst=True
        )
        cross_entropy_shot = F.binary_cross_entropy(
                    pred_shot.cpu(), shot_input[:,1:,:].cpu(), reduction="sum"
                )/(shot_input[:,1:,:].shape[1]*shot_input[:,1:,:].shape[0])

        cross_entropy_hit = F.binary_cross_entropy(
                    pred_hit.cpu(), hit_input[:,1:,:].cpu(), reduction="sum"
                ) /(hit_input[:,1:,:].shape[1]*shot_input[:,1:,:].shape[0])
        
        reg_loss = torch.tensor(0.0, device=diverse_pred_traj.device)
        cal_distance = diverse_pred_traj.reshape(batch, num_agents,diverse_pred_traj.shape[1],diverse_pred_traj.shape[2])
        reg_loss += torch.norm(cal_distance[:,0,:,:] - cal_distance[:,1,:,:] , p=2)
        reg_loss += torch.norm(cal_distance[:,1,:,:] - cal_distance[:,2,:,:] , p=2)
        reg_loss /= (cal_distance.shape[0]*cal_distance.shape[2])

        total_loss = loss_pred + loss_kl + cross_entropy_shot *0.01+ cross_entropy_hit*0.01 + reg_loss
        # total_loss = loss_pred + loss_kl + loss_recover

        # return total_loss, loss_pred.item(), loss_kl.item(), loss_ade, lose_fde, lose_mse
        return (
            total_loss,
            loss_pred.item(),
            loss_kl.item(),
            loss_ade,
            lose_fde,
            lose_mse,
            cross_entropy_shot,
            cross_entropy_hit,
        )

    def sample(
        self,
        traj: Tensor,
        traj_rel: Tensor,
        group: Tensor,
        shot_type_ohe: Tensor,
        hit_player: Tensor,
        direction:Tensor,
        args,
    ):
        batch, timesteps, num_agents, features = traj.shape
        traj_with_emb = []
        for i in range(num_agents):
            out, _ = self.position_emb(traj[:, :, i, :])
            traj_with_emb.append(out.squeeze(1))
        traj_with_emb = torch.stack(traj_with_emb)
        
        ##### add positional encoding ######
        pos_input = traj
        # input_with_pe = self.positional_encoding(pos_input, 8).clone().detach().to(args.device)


        add_emb = []
        for i, player_emb in enumerate(traj_with_emb):
            new_coords = torch.concat((pos_input[:, :, i, :], player_emb), dim=-1)
            add_emb.append(new_coords)
        # x_emb = torch.stack(add_emb).permute(0, 2, 1, 3).reshape(-1, timesteps, 2 + 1)
        x_emb = torch.stack(add_emb).permute(1, 0, 2, 3)
        x_emb = x_emb.reshape(-1, x_emb.shape[2], x_emb.shape[3]) # batch *num_agents , time, coor

       
        #### add shot type & coordinate encoding ######
        
        shot_with_emb = []
        traj_with_emb = []
        for i in range(num_agents):
            shot_out, locs_out, _, _ = self.shot_type_emb(traj[:, :, i, :], shot_type_ohe[: ,: ,i ,: ])
            shot_with_emb.append(shot_out.squeeze(1))
            traj_with_emb.append(locs_out.squeeze(1))
        traj_with_emb = torch.stack(traj_with_emb).permute(1,0,2,3)
        traj_with_emb = traj_with_emb.reshape(-1, traj_with_emb.shape[2], traj_with_emb.shape[3])
        shot_with_emb = torch.stack(shot_with_emb).permute(1,0,2,3)
        shot_with_emb =shot_with_emb.reshape(-1, shot_with_emb.shape[2], shot_with_emb.shape[3])

        shot_input = (
            shot_type_ohe.permute(0, 2, 1, 3)
            .reshape(-1, timesteps, args.s_dim)
            .to(self.device)
        )

        hit_input = (
            hit_player.permute(0, 2, 1, 3)
            .reshape(-1, timesteps, args.p_dim)
            .to(self.device)
        )
        
        
        traj_rel = traj_rel.permute(0, 2, 1, 3)
        traj_rel = traj_rel.reshape(-1, traj_rel.shape[2], traj_rel.shape[3])
        # emb_feats = torch.concat((x_emb, traj_rel), dim=-1)
        # feats_pos = self.positional_encoding(emb_feats.reshape(-1,num_agents,self.obs_len+self.pred_len,5), 8).clone().detach().to(args.device)
        # feats_pos = feats_pos.reshape(-1,self.obs_len+self.pred_len, feats_pos.shape[-1])
        emb_feats = torch.concat((x_emb[:,:,:2], traj_with_emb, traj_rel, shot_input, shot_with_emb, hit_input), dim=-1)
        feats_pos = self.positional_encoding(emb_feats.reshape(-1,num_agents,self.obs_len+self.pred_len, 26), 8).clone().detach().to(args.device)
        feats_pos = feats_pos.reshape(-1,self.obs_len+self.pred_len, feats_pos.shape[-1])
        real_traj = traj.permute(0, 2, 1, 3)
        real_traj = real_traj.reshape(-1, real_traj.shape[2],  real_traj.shape[3])


        feats = torch.concat((real_traj, traj_rel), dim=-1)
        past_traj = real_traj[:, : self.obs_len, :]
       
       

        ## emb_position + velocity
        past_feats_emb = feats_pos[:, : self.obs_len, :].float() 
        


        direction_input = (
            direction.reshape(timesteps, -1, args.num_agents, 2)
            .permute(1, 2, 0, 3)
            .reshape(-1, timesteps, 2)
            .to(self.device)
        )

        cur_location = feats[:, self.obs_len - 1]
        cur_shot = shot_input[:, self.obs_len - 1]
        cur_hit = hit_input[:, self.obs_len - 1]


        h = torch.zeros(self.n_layers, batch, self.rnn_dim).to(self.device)

        ########################## CVAE encoder ##########################
        past_direction = direction_input[:, : self.obs_len, :]
        # past_feats_emb = torch.concat((past_feats_emb),dim=-1)
        past_feats_emb = past_feats_emb
        # import ipdb; ipdb.set_trace()
        past_feature = self.past_encoder(
            past_feats_emb,
            batch,
            args.num_agents,
        )


        prior_input = past_feature

        sample_num = args.num_samples
        if args.learn_prior:
            past_feature_repeat = prior_input.unsqueeze(0).repeat_interleave(sample_num, dim=0)
            p_z_params = self.pz_layer(past_feature_repeat)
            if args.ztype == "gaussian":
                pz_distribution = Normal(params=p_z_params)
            else:
                ValueError("Unknown hidden distribution!")
        else:
            past_feature_repeat = prior_input.unsqueeze(0).repeat_interleave(sample_num, dim=0)
            if args.ztype == "gaussian":
                pz_distribution = Normal(
                    mu=torch.zeros(past_feature_repeat.shape[0], args.zdim).to(
                        past_traj.device
                    ),
                    logvar=torch.zeros(past_feature_repeat.shape[0], args.zdim).to(
                        past_traj.device
                    ),
                )
            else:
                ValueError("Unknown hidden distribution!")

        pz_sampled = pz_distribution.rsample()
        z = pz_sampled
        pred_traj = []
        pred_shots = []
        pred_hits = []
        for sample_idx, samle_z in enumerate(z):
            diverse_pred_traj, pred_shot, pred_hit= self.decoder(
                past_feature_repeat,
                samle_z,
                batch,
                args.num_agents,
                past_traj,
                cur_location,
                cur_shot, 
                cur_hit,
                sample_num=1,
                mode="inference",
                forcast=True
                )
            
            # diverse_pred_traj = diverse_pred_traj.permute(1, 0, 2, 3)
            pred_traj.append(diverse_pred_traj)
            pred_shots.append(pred_shot)
            pred_hits.append(pred_hit)
        pred_traj = torch.stack(pred_traj)    
        pred_shots = torch.stack(pred_shots)    
        pred_hits = torch.stack(pred_hits)    

        return pred_traj, pred_shots, pred_hits
