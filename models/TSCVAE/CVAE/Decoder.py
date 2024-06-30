import torch
from torch import nn
from ..TCN.tcn_layer import TemporalConvNet
from ...ConvAutoEncoder.gat_layer import GATLayer
import dgl
import random
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class DecomposeBlock(nn.Module):

    def __init__(self, past_len, future_len, input_dim):
        super(DecomposeBlock, self).__init__()
        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        self.past_len = past_len
        self.future_len = future_len

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
      
  
        self.relu = nn.ReLU()
        self.gat = GATLayer(input_dim+4, input_dim*2)
        self.tcn = TemporalConvNet(
            input_dim +4,
            [16, 16, 32],
        )
         

    
    def create_fully_connected_dgl_graph(self, num_nodes):
     
        src, dst = zip(
            *[(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        )
        g = dgl.graph((src, dst))
        return g

    def batch_fully_connected_graphs(self, batch_size, num_nodes):
 
        graphs = [
            self.create_fully_connected_dgl_graph(num_nodes) for _ in range(batch_size)
        ]
        batched_graph = dgl.batch(graphs)
        return batched_graph

    def forward(
        self,
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
               
        y_hat_list = torch.stack(y_hat_list).permute(1,0,2)[:,:,:2]
        s_hat_list = torch.stack(s_hat_list).permute(1,0,2)
        p_hat_list = torch.stack(p_hat_list).permute(1,0,2)
    

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

     
        z_in = z.view(-1, sample_num, z.shape[-1])

        hidden = z_in
        
        hidden = hidden.view(agent_num * sample_num, -1)
       
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
      
        norm_seq = prediction.view(agent_num * sample_num, -1, 2)
        norm_shot = prediction_shot.view(agent_num * sample_num, -1, 16)
        norm_hit = prediction_hit.view(agent_num * sample_num, -1, 4)

  
        
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
       
        return out_seq, norm_shot, norm_hit


