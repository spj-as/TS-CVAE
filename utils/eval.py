import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
import matplotlib.patches as mpatches


def one_hot_encode(inds, N):
    dims = [inds.size(i) for i in range(len(inds.size()))]
    inds = inds.unsqueeze(-1).long()
    dims.append(N)
    ret = (torch.zeros(dims)).cuda()
    ret.scatter_(-1, inds, 1)
    return ret


def sample_multinomial(probs):
    """Each element of probs tensor [shape = (batch, s_dim)] has 's_dim' probabilities (one for each grid cell),
    i.e. is a row containing a probability distribution for the goal. We sample n (=batch) indices (one for each row)
    from these distributions, and covert it to a 1-hot encoding."""
    # probs = probs / probs.sum(dim=1, keepdim=True)
    assert (probs >= 0).all(), "Negative probabilities found"
    assert not torch.isnan(probs).any(), "NaN values found"
    assert not torch.isinf(probs).any(), "Infinite values found"
    inds = torch.multinomial(probs, 1).data.long().squeeze()
    ret = one_hot_encode(inds, probs.size(-1))
    return ret


def average_displacement_error(
    pred: torch.Tensor,
    actual: torch.Tensor,
    pred_len: int = 2,
    player_num: int = 4,
    skipFirst=False,
) -> torch.Tensor:
    """
    Average Displacement Error

    Parameters
    ----------
    pred : torch.Tensor
        [batch_size, seq_len * player_num, input_size]
    actual : torch.Tensor
        [batch_size, seq_len * player_num, input_size]

    Returns
    -------
    torch.Tensor
        Return loss
    """

    pred_ = pred.reshape(-1, player_num, pred_len, 2)
    if skipFirst:
        actual_ = actual.reshape(-1, player_num, pred_len + 1, 2)[:, :, 1:, :]
    else:
        actual_ = actual.reshape(-1, player_num, pred_len, 2)

    batch_size, _, pred_len, _ = pred_.shape  # pred_ shape (batch_size, pred_len, 4, 2)

    total_ADE = 0.0
    for batch in range(batch_size):
        ade_numerator = 0.0
        ade_denominator = 0

        for step in range(pred_len):
            mask_val = (actual_[batch, 0, step, 0] != 0) and (
                actual_[batch, 0, step, 1] != 0
            )
            if mask_val:
                ade_denominator += 1

                err_step = (pred_[batch, :, step, :] - actual_[batch, :, step, :]) ** 2
                err_sqrt_sum = torch.sum(
                    torch.sqrt(torch.sum(err_step, dim=-1)), dim=-1
                )
                if err_sqrt_sum != err_sqrt_sum:
                    err_sqrt_sum = 0.0

                ade_numerator += err_sqrt_sum
        if ade_denominator > 0:
            ade_batch = ade_numerator / float(ade_denominator)
            total_ADE += ade_batch
    if batch_size > 0:
        ADE = total_ADE / batch_size
        ADE = ADE / player_num
    return ADE
    # print("Average Displacement Error:", ADE)

  




def average_displacement_error_cal(
    pred: torch.Tensor,
    actual: torch.Tensor,
    pred_len: int = 5,
    player_num: int = 4,
    skipFirst=False,
) -> torch.Tensor:
   
    pred_ = pred.reshape(-1, player_num, pred_len, 2)
    if skipFirst:
        actual_ = actual.reshape(-1, player_num, pred_len + 1, 2)[:, :, 1:, :]
    else:
        actual_ = actual.reshape(-1, player_num, pred_len, 2)

    batch_size, _, pred_len, _ = pred_.shape  # pred_ shape (batch_size, pred_len, 4, 2)

    total_ADEs = {i: 0.0 for i in range(pred_len)}
    for step_len in range(1, pred_len + 1):
        ade_denominator = batch_size * player_num * step_len

        err_step = (pred_[:, :, :step_len, :] - actual_[:, :, :step_len, :]) ** 2
        err_sqrt_sum = torch.sum(torch.sqrt(torch.sum(err_step, dim=-1)), dim=-1).sum(-1).sum(-1)
    
        total_ADEs[step_len - 1] = err_sqrt_sum / ade_denominator

    # Calculate the average ADE for each prediction length over all batches
    average_ADEs = {step_len: total_ADEs[step_len] for step_len in range(pred_len)}

    return average_ADEs[0].item(), average_ADEs[1].item(), average_ADEs[2].item(), average_ADEs[3].item(), average_ADEs[4].item()
 


def final_displacement_error(
    pred: torch.Tensor,
    actual: torch.Tensor,
    player_num: int = 4,
    pred_len: int = 1,
    skipFirst=False,
) -> torch.Tensor:
    """
    Final Displacement Error

    Parameters
    ----------
    pred : torch.Tensor
        [batch_size, seq_len, player_num, input_size]
    actual : torch.Tensor
        [batch_size, seq_len, player_num, input_size]
    player_num : int, optional
        Number of players, by default 4

    Returns
    -------
    torch.Tensor
        Return loss and batch_size
    """
    pred_ = pred.reshape(-1, player_num, pred_len, 2)
    if skipFirst:
        actual_ = actual.reshape(-1, player_num, pred_len + 1, 2)[:, :, 1:, :]
    else:
        actual_ = actual.reshape(-1, player_num, pred_len, 2)

    batch_size, num_players, pred_len, _ = (
        pred_.shape
    )  # pred_ shape (batch_size, pred_len, num_players, 2)
    total_FDE = 0.0
    for batch in range(batch_size):
        fde_numerator = 0.0
        for player in range(num_players):
            last_idx = -1
            for step in range(pred_len):
                if (actual_[batch, player, step, 0] != 0) and (
                    actual_[batch, player, step, 1] != 0
                ):
                    last_idx = step

            if last_idx == -1:
                continue

            err_last_step = torch.sum(
                (
                    pred_[batch, player, last_idx, :]
                    - actual_[batch, player, last_idx, :]
                )
                ** 2
            )

            fde_sqrt = torch.sqrt(err_last_step)
            if fde_sqrt != fde_sqrt:
                fde_sqrt = 0.0

            fde_numerator += fde_sqrt
        total_FDE += fde_numerator

    if batch_size > 0:
        FDE = total_FDE / batch_size
        FDE /= num_players

    return FDE


def mean_square_error(
    pred: torch.Tensor,
    actual: torch.Tensor,
    pred_len: int = 1,
    player_num: int = 4,
    skipFirst=False,
) -> torch.Tensor:
    pred_ = pred.reshape(-1, player_num, pred_len, 2)
    if skipFirst:
        actual_ = actual.reshape(-1, player_num, pred_len + 1, 2)[:, :, 1:, :]
    else:
        actual_ = actual.reshape(-1, player_num, pred_len, 2)

    batch_size, _, num_steps, _ = actual_.shape
    MSE = 0.0

    for batch in range(batch_size):
        mse_denominator = 0.0
        mse_numerator = 0.0
        for step in range(num_steps):
            mask_val = (actual_[batch, 0, step, 0] != 0) and (
                actual_[batch, 0, step, 1] != 0
            )
            if mask_val:
                mse_denominator += 1.0
                err = (pred_[batch, :, step, :] - actual_[batch, :, step, :]) ** 2
                err_sum = torch.sum(err, dim=-1)
                err_sum = torch.sum(err_sum, dim=-1)
                if err_sum != err_sum:
                    err_sum = 0.0
                mse_numerator += err_sum
        if mse_denominator > 0:
            mse_value = mse_numerator / mse_denominator
            MSE += mse_value

    if batch_size > 0:
        MSE = MSE / batch_size
        MSE /= player_num

    return MSE


def scale_value_x(val, old_min, old_max, new_min=1, new_max=-1):
    return ((val - old_min) * (new_max - new_min) / (old_max - old_min)) + new_min


def scale_value_y(val, old_min, old_max, new_min=0, new_max=-1):
    return ((val - old_min) * (new_max - new_min) / (old_max - old_min)) + new_min

def scale_points(points, old_max_x, old_max_y):
    return (
        [scale_value_x(x, 0, old_max_x) for x in points[0]],
        [scale_value_y(y, 0, old_max_y) for y in points[1]],
    )


def plot(
    gt_loc: torch.Tensor,
    pred_loc: torch.Tensor,
    folder: Path,
    epoch: int,
    batch_idx: int,
    pred_len: int,
):

    pred_loc_reshape = pred_loc
    gt_loc_reshape = gt_loc
    pred_loc_reshape = pred_loc_reshape.reshape(-1, 4, pred_len, 2)
    gt_loc_reshape = gt_loc_reshape.reshape(-1, 4, pred_len, 2)
    batch_size, num_players, time_step, _ = pred_loc_reshape.shape
    colors = plt.cm.jet(np.linspace(0, 1, num_players))
    # colors = ["red", "yellow", "cyan", "blue"] 
    offset = 0.02  # offset to reduce overlapping
    for idx in range(batch_size):

        fig, ax = plt.subplots(figsize=(16, 4))
        ax.set_xlim(1, -1)
        ax.set_ylim(0, -1)
        ax.set_aspect("equal")

        # Scale and plot the values
        x_vals = [0, 0, 134, 134, 0]
        y_vals = [0, 61, 61, 0, 0]
        ax.plot(*scale_points((x_vals, y_vals), 134, 61))

        ax.plot(
            *scale_points(
                ([0, 0, 134, 134, 0], [(4.2), (61 - 4.2), (61 - 4.2), (4.2), (4.2)]),
                134,
                61,
            ),
            "k--",
        )

        ax.plot(*scale_points(([0, 134 / 2 - 19.8], [61 / 2, 61 / 2]), 134, 61), "k--")
        ax.plot(
            *scale_points(([134 / 2 + 19.8, 134], [61 / 2, 61 / 2]), 134, 61), "k--"
        )
        ax.plot(
            *scale_points(([134 / 2 - 19.8, 134 / 2 - 19.8], [0, 61]), 134, 61), "k--"
        )
        ax.plot(
            *scale_points(([134 / 2 + 19.8, 134 / 2 + 19.8], [0, 61]), 134, 61), "k--"
        )
        ax.plot(*scale_points(([7.2, 7.2], [0, 61]), 134, 61), "k--")
        ax.plot(*scale_points(([134 - 7.2, 134 - 7.2], [0, 61]), 134, 61), "k--")
        ax.plot(*scale_points(([134 / 2, 134 / 2], [0, 61]), 134, 61), "r-")

        name = []
        for player in range(num_players):
            pred_ = pred_loc_reshape[idx, player, :, :].cpu()
            gt_ = gt_loc_reshape[idx, player, :, :].cpu()

            gt_mask = (gt_[:, 1] != 0) & (gt_[:, 0] != 0)

            (pred_line,) = ax.plot(
                pred_[gt_mask, 1].cpu() + offset * player,
                pred_[gt_mask, 0].cpu() + offset * player,
                color=colors[player],
                linestyle="-",
                marker="o",
                markersize=10,
                label=f"Predicted",
            )

            (gt_line,) = ax.plot(
                gt_[gt_mask, 1].cpu() + offset * player,
                gt_[gt_mask, 0].cpu() + offset * player,
                color=colors[player],
                linestyle="--",
                marker="x",
                markersize=10,
                label=f"Ground Truth",
            )

            if player == 0:
                name.append(pred_line)
                name.append(gt_line)

            for t in range(time_step):
                ax.text(
                    pred_[t, 1].cpu() + offset,
                    pred_[t, 0].cpu() + offset,
                    str(t),
                    fontsize=9,
                    ha="right",
                    va="bottom",
                )
                ax.text(
                    gt_[t, 1].cpu() + offset,
                    gt_[t, 0].cpu() + offset,
                    str(t),
                    fontsize=9,
                    ha="right",
                    va="bottom",
                )
            ax.legend(handles=name, loc="upper left", bbox_to_anchor=(1, 1))

        parent = folder / f"epoch_{epoch}"
        parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(parent / f"{batch_idx}_{idx}.png", bbox_inches="tight")
        plt.close(fig)


def plot_left(
    gt_loc: torch.Tensor,
    pred_loc: torch.Tensor,
    folder: Path,
    epoch: int,
    batch_idx: int,
    pred_len: int,
):

    pred_loc_reshape = pred_loc
    gt_loc_reshape = gt_loc
    pred_loc_reshape = pred_loc_reshape.reshape(-1, 4, pred_len, 2)
    gt_loc_reshape = gt_loc_reshape.reshape(-1, 4, pred_len, 2)
    batch_size, num_players, time_step, _ = pred_loc_reshape.shape
    colors = plt.cm.jet(np.linspace(0, 1, num_players))
    # colors = ["red", "yellow", "cyan", "blue"] 
    offset = 0.02  # offset to reduce overlapping
    for idx in range(batch_size):

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(1, 0)
        ax.set_ylim(0, -1)
        ax.set_aspect("equal")

        # Scale and plot the values
        x_vals = [0, 0, 134, 134, 0]
        y_vals = [0, 61, 61, 0, 0]
        ax.plot(*scale_points((x_vals, y_vals), 134, 61))

        ax.plot(
            *scale_points(
                ([0, 0, 134, 134, 0], [(4.2), (61 - 4.2), (61 - 4.2), (4.2), (4.2)]),
                134,
                61,
            ),
            "k--",
        )

        ax.plot(*scale_points(([0, 134 / 2 - 19.8], [61 / 2, 61 / 2]), 134, 61), "k--")
        ax.plot(
            *scale_points(([134 / 2 + 19.8, 134], [61 / 2, 61 / 2]), 134, 61), "k--"
        )
        ax.plot(
            *scale_points(([134 / 2 - 19.8, 134 / 2 - 19.8], [0, 61]), 134, 61), "k--"
        )
        ax.plot(
            *scale_points(([134 / 2 + 19.8, 134 / 2 + 19.8], [0, 61]), 134, 61), "k--"
        )
        ax.plot(*scale_points(([7.2, 7.2], [0, 61]), 134, 61), "k--")
        ax.plot(*scale_points(([134 - 7.2, 134 - 7.2], [0, 61]), 134, 61), "k--")
        ax.plot(*scale_points(([134 / 2, 134 / 2], [0, 61]), 134, 61), "r-")
        
        name = []
        for player in range(num_players):
            pred_ = pred_loc_reshape[idx, player, :, :].cpu()
            gt_ = gt_loc_reshape[idx, player, :, :].cpu()
            if(gt_[0,1] >= 0 or gt_[1,1] >= 0):
                gt_mask = (gt_[:, 1] != 0) & (gt_[:, 0] != 0)

                (pred_line,) = ax.plot(
                    pred_[gt_mask, 1].cpu() + offset * player,
                    pred_[gt_mask, 0].cpu() + offset * player,
                    color=colors[player],
                    linestyle="-",
                    marker="o",
                    markersize=10,
                    label=f"Predicted",
                )

                (gt_line,) = ax.plot(
                    gt_[gt_mask, 1].cpu() + offset * player,
                    gt_[gt_mask, 0].cpu() + offset * player,
                    color=colors[player],
                    linestyle="--",
                    marker="x",
                    markersize=10,
                    label=f"Ground Truth",
                )

                if player == 0 or player ==2:
                    name.append(pred_line)
                    name.append(gt_line)

                for t in range(time_step):
                    ax.text(
                        pred_[t, 1].cpu() + offset,
                        pred_[t, 0].cpu() + offset,
                        str(t),
                        fontsize=9,
                        ha="right",
                        va="bottom",
                    )
                    ax.text(
                        gt_[t, 1].cpu() + offset,
                        gt_[t, 0].cpu() + offset,
                        str(t),
                        fontsize=9,
                        ha="right",
                        va="bottom",
                    )
                ax.legend(handles=name, loc="upper left", bbox_to_anchor=(1, 1))

        parent = folder / f"epoch_{epoch}/left"
        parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(parent / f"{batch_idx}_{idx}.png", bbox_inches="tight")
        plt.close(fig)

def plot_right(
    gt_loc: torch.Tensor,
    pred_loc: torch.Tensor,
    folder: Path,
    epoch: int,
    batch_idx: int,
    pred_len: int,
):

    pred_loc_reshape = pred_loc
    gt_loc_reshape = gt_loc
    pred_loc_reshape = pred_loc_reshape.reshape(-1, 4, pred_len, 2)
    gt_loc_reshape = gt_loc_reshape.reshape(-1, 4, pred_len, 2)
    batch_size, num_players, time_step, _ = pred_loc_reshape.shape
    colors = plt.cm.jet(np.linspace(0, 1, num_players))
    # colors = ["red", "yellow", "cyan", "blue"] 
    offset = 0.02  # offset to reduce overlapping
    for idx in range(batch_size):

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1, 0)
        ax.set_ylim(0, -1)
        ax.set_aspect("equal")

        # Scale and plot the values
        x_vals = [0, 0, 134, 134, 0]
        y_vals = [0, 61, 61, 0, 0]
        ax.plot(*scale_points((x_vals, y_vals), 134, 61))

        ax.plot(
            *scale_points(
                ([0, 0, 134, 134, 0], [(4.2), (61 - 4.2), (61 - 4.2), (4.2), (4.2)]),
                134,
                61,
            ),
            "k--",
        )

        ax.plot(*scale_points(([0, 134 / 2 - 19.8], [61 / 2, 61 / 2]), 134, 61), "k--")
        ax.plot(
            *scale_points(([134 / 2 + 19.8, 134], [61 / 2, 61 / 2]), 134, 61), "k--"
        )
        ax.plot(
            *scale_points(([134 / 2 - 19.8, 134 / 2 - 19.8], [0, 61]), 134, 61), "k--"
        )
        ax.plot(
            *scale_points(([134 / 2 + 19.8, 134 / 2 + 19.8], [0, 61]), 134, 61), "k--"
        )
        ax.plot(*scale_points(([7.2, 7.2], [0, 61]), 134, 61), "k--")
        ax.plot(*scale_points(([134 - 7.2, 134 - 7.2], [0, 61]), 134, 61), "k--")
        ax.plot(*scale_points(([134 / 2, 134 / 2], [0, 61]), 134, 61), "r-")
        
        name = []
        for player in range(num_players):
            pred_ = pred_loc_reshape[idx, player, :, :].cpu()
            gt_ = gt_loc_reshape[idx, player, :, :].cpu()
            if(gt_[0,1] < 0 or gt_[1,1] < 0):
                gt_mask = (gt_[:, 1] != 0) & (gt_[:, 0] != 0)

                (pred_line,) = ax.plot(
                    pred_[gt_mask, 1].cpu() + offset * player,
                    pred_[gt_mask, 0].cpu() + offset * player,
                    color=colors[player],
                    linestyle="-",
                    marker="o",
                    markersize=10,
                    label=f"Predicted",
                )

                (gt_line,) = ax.plot(
                    gt_[gt_mask, 1].cpu() + offset * player,
                    gt_[gt_mask, 0].cpu() + offset * player,
                    color=colors[player],
                    linestyle="--",
                    marker="x",
                    markersize=10,
                    label=f"Ground Truth",
                )

                if player == 0 or player ==2:
                    name.append(pred_line)
                    name.append(gt_line)

                for t in range(time_step):
                    ax.text(
                        pred_[t, 1].cpu() + offset,
                        pred_[t, 0].cpu() + offset,
                        str(t),
                        fontsize=9,
                        ha="right",
                        va="bottom",
                    )
                    ax.text(
                        gt_[t, 1].cpu() + offset,
                        gt_[t, 0].cpu() + offset,
                        str(t),
                        fontsize=9,
                        ha="right",
                        va="bottom",
                    )
                ax.legend(handles=name, loc="upper left", bbox_to_anchor=(1, 1))
                # custom_ticks = [10, 20, 30, 40, 50]  # Specify your custom tick values
                custom_labels = [1.0, 0.8, 0.6, 0.4, 0.2, 0]  # Specify your custom tick labels

                # ax.set_xticks(custom_ticks)
                ax.set_xticklabels(custom_labels)

        parent = folder / f"epoch_{epoch}/right"
        parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(parent / f"{batch_idx}_{idx}.png", bbox_inches="tight")
        plt.close(fig)



