import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from os.path import join
from utils.seed import set_seed
from utils.dataset import get_badminton_datasets
from utils.checkpoint import save_checkpoint
from utils.eval import  average_displacement_error, final_displacement_error, mean_square_error, plot, plot_left, plot_right, average_displacement_error_cal
from models.TSCVAE.tscvae import TSCVAE
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from time import time
from utils.annealing import KLAnnealer
from torch.optim import lr_scheduler
import numpy as np
def tscvae_cli(parser: ArgumentParser) -> ArgumentParser:
    # Required
    parser.add_argument("--name", type=str, required=True, help="current run name")  # fmt: skip

    # Pretrained Options
    parser.add_argument("--position_emb_path", type=str, default="./models/ConvAutoEncoder/weights/best_model.pth", help="path to pretrained position embedding")  # fmt: skip
    parser.add_argument("--shot_emb_path", type=str, default="./models/ShotTypeEmb/weights/best_model.pth", help="path to pretrained shot type")  # fmt: skip

    # Dataset Options
    parser.add_argument("--root", type=str, default=join("data", "badminton", "doubles"), help="root directory of dataset")  # fmt: skip
    parser.add_argument("--obs_len", type=int, default=10, help="observation length")  # fmt: skip
    parser.add_argument("--pred_len", type=int, default=2, help="prediction length")  # fmt: skip
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")  # fmt: skip
    parser.add_argument("--num_agents", type=int, default=4, help="number of workers")  # fmt: skip

    # Optimizer Options
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')  # fmt: skip
    parser.add_argument("-e", "--epochs", type=int, default=500, help="number of epochs")  # fmt: skip

    # Model Options
    parser.add_argument("--clip", type=float, default=3.0, help="gradient clip")  # fmt: skip
    parser.add_argument("--s_dim", type=int, default=16, help="number of shot types")  # fmt: skip
    parser.add_argument("--p_dim", type=int, default=4, help="number of players")  # fmt: skip
    parser.add_argument("--hidden_dim", type=int, default=8, help="hidden dimension")  # fmt: skip
    parser.add_argument("--n_layers", type=int, default=2, help="number of rnn layers")  # fmt: skip
    parser.add_argument("--x_dim", default=2, type=int, help="feature dimension of single agent")  # fmt: skip
    parser.add_argument("--z_dim", default=16, type=int, help="latent dimension")  # fmt: skip
    parser.add_argument("--rnn_dim", default=16, type=int, help="rnn hidden dimension")  # fmt: skip
    parser.add_argument("--resume", action="store_true", default=False, help="resume training from checkpoint")  # fmt: skip
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path")  # fmt: skip
    parser.add_argument("--ce_weight", default=0.01, type=float, required=False, help="Cross-entropy loss weight")


    parser.add_argument('--learn_prior', action='store_true', default=True)
    parser.add_argument('--decay_step', type=int, default=10)
    parser.add_argument('--decay_gamma', type=float, default=0.5)
    parser.add_argument('--iternum_print', type=int, default=100)
    parser.add_argument('--ztype', default='gaussian')
    parser.add_argument('--zdim', type=int, default=16)
    parser.add_argument('--hyper_scales', nargs='+', type=int, default=[1])
    parser.add_argument('--num_decompose', type=int, default=1)


    # Model Options - Graph
    # parser.add_argument("--graph_model", type=str, required=True, choices=["gat", "gcn"], help="graph model")  # fmt: skip
    parser.add_argument("--graph_hid", type=int, default=6, help="graph hidden dimension")  # fmt: skip
    parser.add_argument("--sigma", type=float, default=1.2, help="Sigma value for similarity matrix")  # fmt: skip
    parser.add_argument("--adjacency_type", type=int, default=1, choices=[0, 1, 2], help="Type of adjacency matrix:\n0 (fully connected)\n1 (distances similarity)\n2 (knn similarity)")  # fmt: skip
    parser.add_argument("--top_k_neigh", type=int, default=None)

    # Model Options - Graph (GAT)
    parser.add_argument("--n_heads", type=int, default=2, help="number of heads for gat")  # fmt: skip
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha for leaky relu")  # fmt: skip

    # Model Options - Teacher Forcing
    parser.add_argument("--tf_disable", action="store_true", default=False, help="disable teacher forcing")  # fmt: skip
    parser.add_argument("--tf_threshold", type=float, default=1.0, help="initial teacher forcing threshold")  # fmt: skip

    # Kl annealing strategy arguments
    parser.add_argument("--kl_anneal_type", type=str, default="Cyclical", help="")
    parser.add_argument("--kl_anneal_cycle", type=int, default=20, help="")
    parser.add_argument("--kl_anneal_ratio", type=float, default=1, help="")

    # Miscellaneous
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="mode")  # fmt: skip
    parser.add_argument("--seed", type=int, default=123, help="random seed")  # fmt: skip
    parser.add_argument("--device", type=str, default="cuda", help="device")  # fmt: skip
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")  # fmt: skip
    parser.add_argument("--save_every", type=int, default=100, help="save every")  # fmt: skip
    parser.add_argument("--eval_every", type=int, default=1, help="evaluate every")  # fmt: skip

    parser.add_argument("--results_dir", type=str, default="results", help="results directory")  # fmt: skip
    parser.add_argument("--num_samples", default=20, type=int, help="Number of samples for evaluation")  # fmt: skip
    return parser


def train(
    args: Namespace,
    epoch: int,
    model: TSCVAE,
    loader: DataLoader,
    writer: SummaryWriter,
    optimizer: optim.Optimizer,
    beta: float,
    tf_threshold: float,
    scheduler:optim.lr_scheduler
):
    start_time = time()

    losses: list[torch.Tensor] = []
    klds: list[torch.Tensor] = []
    nlls: list[torch.Tensor] = []
    ce_shot: list[torch.Tensor] = []
    ce_hit: list[torch.Tensor] = []

    ades: list[torch.Tensor] = []
    fdes: list[torch.Tensor] = []
    mses: list[torch.Tensor] = []

    model.train()
    for batch_idx, data in enumerate(loader):
        data = [tensor.to(args.device) for tensor in data]
        (
            all_traj,
            all_traj_rel,
            all_goals_ohe,
            all_hit,
        ) = data

      
    

        total_loss, loss_pred, loss_kl, loss_ade, loss_fde, loss_mse, loss_ce_shot, loss_ce_hit = model.forward(
            traj=all_traj,
            traj_rel=all_traj_rel,
            shot_type_ohe=all_goals_ohe,
            hit_player=all_hit,
            tf_threshold=tf_threshold,
            obs_len=args.obs_len,
            args=args,
            epoch = epoch,
            beta = beta,
        )
        # Compute loss
        loss = total_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()

        ades.append(loss_ade)
        fdes.append(loss_fde)
        mses.append(loss_mse)
        losses.append(loss)
        klds.append(torch.tensor(loss_kl))
        nlls.append(torch.tensor(loss_pred))
        ce_shot.append(loss_ce_shot)
        ce_hit.append(loss_ce_hit)

    # Compute mean metrics
    avg_loss = torch.sum(torch.stack(losses)) / len(loader)
    avg_kld = torch.sum(torch.stack(klds)) / len(loader)
    avg_nll = torch.sum(torch.stack(nlls)) / len(loader)
    avg_ce_shot = torch.sum(torch.stack(ce_shot))/ len(loader)

    avg_ce_hit = torch.sum(torch.stack(ce_hit))/ len(loader)


    
    avg_ade = torch.sum(torch.stack(ades)) / len(loader)
    avg_fde = torch.sum(torch.stack(fdes)) / len(loader)
    avg_mse = torch.sum(torch.stack(mses)) / len(loader)
    
    # Log metrics
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/kld", avg_kld, epoch)
    writer.add_scalar("train/nll", avg_nll, epoch)
    writer.add_scalar("train/ade", avg_ade, epoch)
    writer.add_scalar("train/fde", avg_fde, epoch)
    writer.add_scalar("train/mse", avg_mse, epoch)
    writer.add_scalar("train/avg_ce_shot", avg_ce_shot, epoch)
    writer.add_scalar("train/avg_ce_hit", avg_ce_hit, epoch)


    writer.add_scalar("train/beta", beta, epoch)
    writer.add_scalar("train/tf_threshold", tf_threshold, epoch)

    logging.info(f"Epoch {epoch} (Train) | ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f} | MSE: {avg_mse:.4f} | Loss: {avg_loss:.4f} | KLD: {avg_kld:.4f} | NLL: {avg_nll:.4f} | CE: {avg_ce_shot:.4f} | CE Hit: {avg_ce_hit:.4f} ")  # fmt: skip
  
    logging.info("Epoch [{}], time elapsed: {:3f}".format(epoch, time() - start_time))


@torch.no_grad()
def evaluation(
    args: Namespace,
    writer: SummaryWriter,
    epoch: int,
    model: TSCVAE,
    loader: DataLoader,
    pic_dir: Path,
    beta: float,
    tf_threshold: float,

) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Evaluate the model on the test set

    Returns
    -------
    loss: float
        Test loss
    kld : float
        Test KL divergence
    nll : float
        Test Negative log likelihood loss
    mean_ce : float
        Mean cross entropy loss
    mena_ce_hit : float
        Mean cross entropy of hit net
    ade : float
        Average displacement error
    fde : float
        Final displacement error
    mse : float
        Mean squared error
    """
    model.eval()

    losses: list[torch.Tensor] = []
    klds: list[torch.Tensor] = []
    nlls: list[torch.Tensor] = []
    ce_shot: list[torch.Tensor] = []
    ce_hit: list[torch.Tensor] = []

    ades: list[torch.Tensor] = []
    fdes: list[torch.Tensor] = []
    mses: list[torch.Tensor] = []

    ADE1 = []
    ADE2 = []
    ADE3 = []
    ADE4 = []
    ADE5 = []


    for batch_idx, data in enumerate(loader):
        data = [tensor.to(args.device) for tensor in data]
        (
            all_traj,
            all_traj_rel,
            all_goals_ohe,
            all_hit,
        ) = data

      

        min_ade = float("Inf")
        min_fde = None
        min_mse = None
        min_plot_GT = None
        min_plot_pred = None
        min_acc = float("Inf")
        min_precision = None
        min_recall = None
        min_ce_shot = None
        min_ce_hit = None
        
        samples, shot, hit = model.sample(
            traj=all_traj,
            traj_rel=all_traj_rel,
            group=all_group,
            shot_type_ohe=all_goals_ohe,
            hit_player=all_hit,
            direction=all_direction,
            args=args
        )
        
        pred_traj_gt = all_traj.permute(0,2,1,3)
        pred_traj_gt = pred_traj_gt.reshape(-1,pred_traj_gt.shape[2], pred_traj_gt.shape[3])[:,args.obs_len:,:]
        # past_traj_gt = all_traj.permute(0,2,1,3)[:,:,:args.obs_len,:]
        prediction = samples.clone().detach()
        prediction = prediction.reshape(args.num_samples, -1, args.pred_len, 2)
        shot = shot.reshape(args.num_samples, -1, args.pred_len, 16)
        hit = hit.reshape(args.num_samples, -1, args.pred_len, 4)
        shot_gt = (
            all_goals_ohe.permute(0, 2, 1, 3)
            .reshape(-1, args.obs_len + args.pred_len, args.s_dim)
        )[:,args.obs_len:,:]

        hit_gt = (
            all_hit.permute(0, 2, 1, 3)
            .reshape(-1, args.obs_len + args.pred_len, args.p_dim)
        )[:,args.obs_len:,:]
        
        ADE = np.zeros((prediction.shape[0], 5))
        for idx, sample in enumerate (prediction):
            plot_GT = pred_traj_gt
            plot_pred = sample
            ade_ = average_displacement_error(pred=sample, actual=pred_traj_gt, pred_len=args.pred_len)
            # ADE[idx] = average_displacement_error_cal(pred=sample, actual=pred_traj_gt, pred_len=args.pred_len)
            fde_ = final_displacement_error(pred=sample, actual=pred_traj_gt, pred_len=args.pred_len)
            mse_ = mean_square_error(pred=sample, actual=pred_traj_gt, pred_len=args.pred_len)
            cross_entropy_shot = F.binary_cross_entropy(
                    shot[idx], shot_gt, reduction="sum"
                )/(shot_gt.shape[0]*shot_gt.shape[1])
            
            cross_entropy_hit = F.binary_cross_entropy(
                    hit[idx], hit_gt, reduction="sum"
                ) /(hit_gt.shape[0]*hit_gt.shape[1])


            if ade_ < min_ade:
                min_ade = ade_
                min_fde = fde_
                min_mse = mse_
                min_plot_GT = pred_traj_gt
                min_plot_pred = plot_pred
                min_ce_shot = cross_entropy_shot
                min_ce_hit = cross_entropy_hit

        plot(
            gt_loc=min_plot_GT,
            pred_loc=min_plot_pred,
            folder=pic_dir,
            epoch=epoch,
            batch_idx=batch_idx,
            pred_len=args.pred_len,
        )
        plot_left(
            gt_loc=min_plot_GT,
            pred_loc=min_plot_pred,
            folder=pic_dir,
            epoch=epoch,
            batch_idx=batch_idx,
            pred_len=args.pred_len,
        )
        plot_right(
            gt_loc=min_plot_GT,
            pred_loc=min_plot_pred,
            folder=pic_dir,
            epoch=epoch,
            batch_idx=batch_idx,
            pred_len=args.pred_len,
        )
       


        ades.append(min_ade)
        fdes.append(min_fde)
        mses.append(min_mse)
        ce_shot.append(min_ce_shot)
        ce_hit.append(min_ce_hit)
 
    avg_ade = torch.sum(torch.stack(ades)) / len(loader)
    avg_fde = torch.sum(torch.stack(fdes)) / len(loader)
    avg_mse = torch.sum(torch.stack(mses)) / len(loader)
    avg_ce_shot = torch.sum(torch.stack(ce_shot)) / len(loader)
    avg_ce_hit = torch.sum(torch.stack(ce_hit)) / len(loader)


    writer.add_scalar("test/ade", avg_ade, epoch//args.eval_every)
    writer.add_scalar("test/fde", avg_fde, epoch//args.eval_every)
    writer.add_scalar("test/mse", avg_mse, epoch//args.eval_every)
    writer.add_scalar("test/ce_shot", avg_ce_shot, epoch//args.eval_every)
    writer.add_scalar("test/ce_hit", avg_ce_hit, epoch//args.eval_every)
 

    return  avg_ade, avg_fde, avg_mse, avg_ce_shot, avg_ce_hit

def tscvae_main(args: Namespace):
    annealer = KLAnnealer(args)
    g = set_seed(args.seed)

    base = Path(args.results_dir, args.name)
    sw_dir = Path(base, "events")
    log_dir = Path(base, "logs")
    ckpt_dir = Path(base, "checkpoints")
    pic_dir = Path(base, "pics")

    sw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pic_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir.joinpath(f"{time()}.log"),
        filemode="a",
        level=logging.INFO,
        format="[%(levelname)s | %(asctime)s]: %(message)s",
        datefmt="%Y/%m/%d %H:%M",
    )

    train_set, test_set = get_badminton_datasets(
        root=args.root, n_agents=4, obs_len=args.obs_len, pred_len=args.pred_len
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, generator=g)  # fmt: skip
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, generator=g)  # fmt: skip

    n_max_agents = max(train_set.__max_agents__, test_set.__max_agents__)
    tf_threshold = args.tf_threshold
    if args.mode == "train":
        writer = SummaryWriter(str(sw_dir))
        checkpoint: dict[str] = torch.load(args.checkpoint) if args.resume else {}

        model = TSCVAE(args=args, n_max_player=n_max_agents)
        model.load_state_dict(checkpoint["model"]) if args.resume else None
        model = model.to(args.device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint["optimizer"]) if args.resume else None

        start_epoch = checkpoint["epoch"] if args.resume else 0
        best_ade = checkpoint.get("metrics", {}).get("best_ade") if args.resume else float("Inf")
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)

        for epoch in range(start_epoch + 1, args.epochs + 1):
            # Train the model
            beta = annealer.get_beta()

            train(
                args,
                epoch=epoch,
                model=model,
                loader=train_loader,
                writer=writer,
                optimizer=optimizer,
                beta=beta,
                tf_threshold=tf_threshold,
                scheduler =scheduler
            )

            # Preriodically evaluate the model
            if epoch % args.eval_every == 0:
                ade, fde, mse, ce_shot, ce_hit= evaluation(
                    args,
                    writer=writer,
                    epoch=epoch,
                    model=model,
                    loader=test_loader,
                    pic_dir=pic_dir,
                    beta=beta,
                    tf_threshold=tf_threshold,
                )
                logging.info(f"Epoch {epoch} (Val) | ADE: {ade:.4f} | FDE: {fde:.4f} | MSE: {mse:.4f}| CE shot: {ce_shot:.4f}| CE hit: {ce_hit:.4f}  ")  # fmt: skip

                if 0 <= ade < best_ade:
                    best_ade = ade
                    path = ckpt_dir.joinpath(f"best_ade_{best_ade:.4f}_epoch_{epoch}.pth")
                    save_checkpoint(str(path), args, epoch, model, optimizer, {"best_ade": best_ade})

            # Periodically save the model
            if epoch % args.save_every == 0:
                path = ckpt_dir.joinpath(f"epoch_{epoch}.pth")
                save_checkpoint(str(path), args, epoch, model, optimizer, {"best_ade": best_ade})
            tf_threshold *= 0.99
            annealer.update()

        writer.close()

    elif args.mode == "eval":
        assert args.checkpoint is not None, "checkpoint path is required for testing"
        evaluation(args)
