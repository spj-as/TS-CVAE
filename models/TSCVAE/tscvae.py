import torch
from torch import nn, Tensor
import torch.nn.init as init
import torch.nn.functional as F
from argparse import Namespace
from models.ShotTypeEmb.auto_encoder import ShotTypeEmb
from models.ConvAutoEncoder.gat_layer import GATLayer, StandardGAT
from models.TSCVAE.TCN.tcn_layer import TemporalConvNet
from models.TSCVAE.CVAE.allEncoder import FutureEncoder
from models.TSCVAE.CVAE.pastEncoder import PastEncoder
from models.TSCVAE.CVAE.Decoder import TrajDecoder
from models.TSCVAE.CVAE.Distribution import Normal
import numpy as np
import random
import dgl
import random
import matplotlib.pyplot as plt
import seaborn as sns

from utils.eval import (
    mean_square_error,
    average_displacement_error,
    final_displacement_error,
)

class TSCVAE(nn.Module):
    def __init__(self, args: Namespace, n_max_player: int):
        super(TSCVAE, self).__init__()

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

        self.graph_hid: int = args.graph_hid
        self.adjacency_type: str = args.adjacency_type
        self.top_k_neigh: int = args.top_k_neigh
        self.sigma = args.sigma
        self.alpha: float = args.alpha
        self.n_heads: int = args.n_heads
        self.device: str = args.device
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
        self.nn = nn.Linear(24, 8)
      
        self.pz_layer = nn.Linear(944, 2 * args.z_dim)
        self.q_layer = nn.Linear(2 * args.zdim, 2 * args.z_dim)

        # Decoders
        self.decoder = TrajDecoder(args)

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std


    def calculate_loss_pred(self, pred, target, batch_size):
        loss = (target - pred).pow(2).sum()
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


   

    def forward(
        self,
        traj: Tensor,
        traj_rel: Tensor,
        shot_type_ohe: Tensor,
        hit_player: Tensor,
        tf_threshold: float,
        obs_len: int,
        args,
        epoch: int,
        beta: float,
    ):

        batch, timesteps, num_agents, features = traj.shape
        
        traj_with_emb = []
        # for i in range(num_agents):
        #     out, _ = self.position_emb(traj[:, :, i, :])
        #     traj_with_emb.append(out.squeeze(1))
        # traj_with_emb = torch.stack(traj_with_emb)
        
        # ##### add positional encoding ######
        # pos_input = traj


        # add_emb = []
        # for i, player_emb in enumerate(traj):
        #     new_coords = torch.concat((pos_input[:, :, i, :], player_emb), dim=-1)
        #     add_emb.append(new_coords)
        x_emb = traj.permute(0, 2, 1, 3)
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
    
        emb_feats = torch.concat((x_emb, traj_with_emb, traj_rel, shot_input, shot_with_emb, hit_input), dim=-1)
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
        cur_location = feats
        h = torch.zeros(self.n_layers, batch, self.rnn_dim).to(self.device)

        ########################## CVAE encoder ##########################
      
        cur_shot = shot_input

      
        cur_hit = hit_input

        past_feats_emb_shot = past_feats_emb
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
        
        loss_pred = self.calculate_loss_pred(diverse_pred_traj, real_traj[:,1:], batch)
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
        
      
        total_loss = loss_pred + loss_kl + cross_entropy_shot *0.01+ cross_entropy_hit*0.01 
       
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
        shot_type_ohe: Tensor,
        hit_player: Tensor,
        args,
    ):
        batch, timesteps, num_agents, features = traj.shape
        # traj_with_emb = []
        # for i in range(num_agents):
        #     out, _ = self.position_emb(traj[:, :, i, :])
        #     traj_with_emb.append(out.squeeze(1))
        # traj_with_emb = torch.stack(traj_with_emb)
        
        ##### add positional encoding ######
        pos_input = traj


        # add_emb = []
        # for i, player_emb in enumerate(traj_with_emb):
        #     new_coords = torch.concat((pos_input[:, :, i, :], player_emb), dim=-1)
        #     add_emb.append(new_coords)
        x_emb = traj.permute(0, 2, 1, 3)
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
        emb_feats = torch.concat((x_emb[:,:,:], traj_with_emb, traj_rel, shot_input, shot_with_emb, hit_input), dim=-1)
        feats_pos = self.positional_encoding(emb_feats.reshape(-1,num_agents,self.obs_len+self.pred_len, 26), 8).clone().detach().to(args.device)
        feats_pos = feats_pos.reshape(-1,self.obs_len+self.pred_len, feats_pos.shape[-1])
        real_traj = traj.permute(0, 2, 1, 3)
        real_traj = real_traj.reshape(-1, real_traj.shape[2],  real_traj.shape[3])


        feats = torch.concat((real_traj, traj_rel), dim=-1)
        past_traj = real_traj[:, : self.obs_len, :]
       
       

        ## emb_position + velocity
        past_feats_emb = feats_pos[:, : self.obs_len, :].float() 

        cur_location = feats[:, self.obs_len - 1]
        cur_shot = shot_input[:, self.obs_len - 1]
        cur_hit = hit_input[:, self.obs_len - 1]


        h = torch.zeros(self.n_layers, batch, self.rnn_dim).to(self.device)

        ########################## CVAE encoder ##########################
        past_feats_emb = past_feats_emb
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
            
            pred_traj.append(diverse_pred_traj)
            pred_shots.append(pred_shot)
            pred_hits.append(pred_hit)
        pred_traj = torch.stack(pred_traj)    
        pred_shots = torch.stack(pred_shots)    
        pred_hits = torch.stack(pred_hits)    

        return pred_traj, pred_shots, pred_hits
