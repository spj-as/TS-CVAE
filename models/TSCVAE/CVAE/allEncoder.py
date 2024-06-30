import torch
from torch import nn
from ..TCN.tcn_layer import  TemporalConvNet
from ...ConvAutoEncoder.gat_layer import GATLayer, StandardGAT
import dgl

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
            1072 , 2 * self.args.zdim
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
        
        gat_in = inputs.reshape(-1, agent_num, inputs.shape[1], inputs.shape[2])
        team1_feat = gat_in[:, :2, :, :]  
        team2_feat = gat_in[:, 2:, :, :] 
        combined_features = []
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
            
            f_gat_team1 = f_gat_team1.reshape(-1, agent_num //2 , 3, f_gat_team1.shape[1])
            f_gat_team2, att1, att2 = self.relationGAT(team2_graph, team2_graph.ndata['feat'])
            f_gat_team2 = f_gat_team2.reshape(-1, agent_num //2 , 3, f_gat_team2.shape[1])
            combined_features.append(torch.cat([f_gat_team1, f_gat_team2], dim=1))
        combined_features = torch.stack(combined_features)
        combined_features = combined_features.reshape(self.args.obs_len +self.args.pred_len-1, -1, agent_num, 3, combined_features.shape[-1])

        combined_features = combined_features.permute(1, 2, 0, 3, 4)
        combined_features = combined_features.reshape(-1, combined_features.shape[2]*combined_features.shape[3]*combined_features.shape[4])
        

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
        gat_out =gat_out.reshape(
            batch_size * agent_num, -1
        )


        final_feature = torch.cat(
            (gat_out, combined_features), dim=-1
        )  # (batch, agents, feat.)


        final_feature = final_feature.reshape(batch_size * agent_num, -1)
        q_z_params = self.qz_layer(final_feature)

        return q_z_params