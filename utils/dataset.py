import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from os.path import join
from typing import Literal


class BadmintonDataset(Dataset):
    """Dataloder for the Basketball trajectories datasets"""

    def __init__(
        self,
        mode: Literal["train", "test"],
        root: str = None,
        n_agents: int = 5,
        obs_len: int = 10,
        pred_len: int = 2,
    ):
        super(BadmintonDataset, self).__init__()
        assert mode in ["train", "test"], "mode must be either train or test"

        self.data_dir = root or join("data", "badminton", "doubles")
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.n_agents = n_agents

        displacements = np.load(join(self.data_dir, mode, "displacements.npy"), allow_pickle=True)
        velocity = np.load(join(self.data_dir, mode, "velocity.npy"), allow_pickle=True)
        group = np.load(join(self.data_dir, mode, "group.npy"), allow_pickle=True)
        goals = np.load(join(self.data_dir, mode, "goals.npy"), allow_pickle=True)
        hits = np.load(join(self.data_dir, mode, "hit.npy"), allow_pickle=True)
        direction = np.load(join(self.data_dir, mode, "direction.npy"), allow_pickle=True)


        num_seqs = displacements.shape[1] // self.n_agents
        # idxs = [idx for idx in range(0, (num_seqs * self.n_agents) + n_agents, n_agents)]
        # seq_start_end = [[start, end] for start, end in zip(idxs[:], idxs[1:])]
        

        # self.num_samples = len(seq_start_end)

        self.num_samples = displacements.shape[0]

        self.displacements = torch.from_numpy(displacements).type(torch.float)
        self.velocity = torch.from_numpy(velocity).type(torch.float)
        self.group = torch.from_numpy(group).type(torch.float)
        self.goals = torch.from_numpy(goals).type(torch.float)
        self.hits = torch.from_numpy(hits).type(torch.float)
        self.direction = torch.from_numpy(direction).type(torch.float)

    def __len__(self) -> int:
        return self.num_samples

    @property
    def __max_agents__(self) -> int:
        return self.n_agents

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        # start, end = self.seq_start_end[idx]

        out = [
            self.displacements[idx],
            self.velocity[idx],
            self.group[idx],
            self.goals[idx],
            self.hits[idx],
            self.direction[idx],
        ]
        return tuple(out)


def get_badminton_datasets(
    root: str = None,
    n_agents: int = 4,
    obs_len: int = 10,
    pred_len: int = 2,
) -> tuple[BadmintonDataset, BadmintonDataset]:
    train_set = BadmintonDataset(mode="train", root=root, n_agents=n_agents, obs_len=obs_len, pred_len=pred_len)
    test_set = BadmintonDataset(mode="test", root=root, n_agents=n_agents, obs_len=obs_len, pred_len=pred_len)
    return train_set, test_set






class BasketballDataset(Dataset):
    """Dataloder for the Basketball trajectories datasets"""

    def __init__(
        self,
        mode: Literal["train", "test"],
        root: str = None,
        n_agents: int = 10,
        obs_len: int = 30,
        pred_len: int = 20,
    ):
        super(BasketballDataset, self).__init__()
        assert mode in ["train", "test"], "mode must be either train or test"

        self.data_dir = root or join("data", "basketball")
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.n_agents = n_agents

        displacements = np.load(join(self.data_dir, mode, "displacements.npy"), allow_pickle=True)
        velocity = np.load(join(self.data_dir, mode, "velocity.npy"), allow_pickle=True)
      


        self.num_samples = displacements.shape[0]

        self.displacements = torch.from_numpy(displacements).type(torch.float)
        self.velocity = torch.from_numpy(velocity).type(torch.float)
      

    def __len__(self) -> int:
        return self.num_samples

    @property
    def __max_agents__(self) -> int:
        return self.n_agents

    def __getitem__(self, idx: int) -> list[torch.Tensor]:

        out = [
            self.displacements[idx],
            self.velocity[idx],
        ]
        return tuple(out)


def get_basketball_datasets(
    root: str = None,
    n_agents: int = 10,
    obs_len: int = 30,
    pred_len: int = 20,
) -> tuple[BasketballDataset, BasketballDataset]:
    train_set = BasketballDataset(mode="train", root=root, n_agents=n_agents, obs_len=obs_len, pred_len=pred_len)
    test_set = BasketballDataset(mode="test", root=root, n_agents=n_agents, obs_len=obs_len, pred_len=pred_len)
    return train_set, test_set
