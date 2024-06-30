import torch 
from torch import nn
from ..TCN.tcn_layer import TemporalConvNet
from ...ConvAutoEncoder.gat_layer import StandardGAT, GATLayer
import dgl

class PastEncoder(nn.Module):
    def __init__(self, args, in_dim=4):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.scale_number = len(args.hyper_scales)

        self.input_fc = nn.Linear(8, self.model_dim)
        self.input_fc2 = nn.Linear(self.model_dim * args.obs_len, self.model_dim)
        self.input_fc3 = nn.Linear(self.model_dim + 2, self.model_dim)


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
     
        output_feature = final_feature.reshape(
            batch_size * agent_num, -1
        )  # (batch * agents, feat.)
        return output_feature
