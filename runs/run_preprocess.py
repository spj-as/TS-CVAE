from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from os.path import join
from pathlib import Path


class BadmintonPreprocessor:
    def __init__(self, args: Namespace):
        self.args = args

        self.train_path = join(args.root, "train.csv")
        self.test_path = join(args.root, "test.csv")
        self.n_players = args.n_agents

        self.player_loc_cols = [
            "player_A_x",
            "player_A_y",
            "player_B_x",
            "player_B_y",
            "player_C_x",
            "player_C_y",
            "player_D_x",
            "player_D_y",
            # "hit_x",
            # "hit_y",
        ]
        self.hit_loc = ["hit_x", "hit_y"]

        # self.shot_type_categories = ["推撲球", "平球", "網前小球", "發短球", "殺球", "切球", "發長球", "挑球", "長球"]
        shot_type_categories = [['發短球', '放小球', '挑球', '切球', '殺球', '防守回抽', '擋小球', '推撲球', '後場抽平球', '平球', '小平球', '防守回挑', '長球', '過度切球', '發長球', '勾球']]

        self.shot_type_enc = OneHotEncoder(categories=shot_type_categories, handle_unknown='ignore')
        self.hit_player_enc = OneHotEncoder()

    def _preprocess_single(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        displacements = df[self.player_loc_cols].values  # 位移
        nor_max = displacements.max()
        nor_min = displacements.min()
        normalize = (
            lambda data, data_max, data_min: (data - data_min)
            * 2
            / (data_max - data_min)
            - 1
        )
        displacements = normalize(displacements, nor_max, nor_min)
        df[self.player_loc_cols] = displacements

        hit_loc = df[self.hit_loc].values  # 位移
        nor_max = hit_loc.max()
        nor_min = hit_loc.min()
      
        hit_loc = normalize(hit_loc, nor_max, nor_min)
        df[self.hit_loc] = hit_loc

        displacements = df[["player_A_x",
            "player_A_y",
            "player_B_x",
            "player_B_y",
            "player_C_x",
            "player_C_y",
            "player_D_x",
            "player_D_y",
            # "hit_x",
            # "hit_y"
            ]].values 

        
        # velocity
        df[
            [
                "delta_A_x",
                "delta_A_y",
                "delta_B_x",
                "delta_B_y",
                "delta_C_x",
                "delta_C_y",
                "delta_D_x",
                "delta_D_y",
                # "delta_hit_x",
                # "delta_hit_y",
            ]
        ] = df.groupby("rally_id")[
            [
                "player_A_x",
                "player_A_y",
                "player_B_x",
                "player_B_y",
                "player_C_x",
                "player_C_y",
                "player_D_x",
                "player_D_y",
                # "hit_x",
                # "hit_y",
            ]
        ].diff()

        df["delta_time"] = df.groupby("rally_id")["end_frame_num"].shift(
            1
        ) - df.groupby("rally_id")["start_frame_num"].shift(1)
        df.loc[df["delta_time"] == 0, "delta_time"] = 1
        # df["delta_time"] = df.groupby("rally_id")["start_frame_num"].shift(-1) - df["start_frame_num"]
        df["velocity_A_x"] = df["delta_A_x"] 
        df["velocity_A_y"] = df["delta_A_y"] 
        df["velocity_B_x"] = df["delta_B_x"] 
        df["velocity_B_y"] = df["delta_B_y"] 
        df["velocity_C_x"] = df["delta_C_x"] 
        df["velocity_C_y"] = df["delta_C_y"]
        df["velocity_D_x"] = df["delta_D_x"] 
        df["velocity_D_y"] = df["delta_D_y"] 
        # df["velocity_hit_x"] = df["delta_hit_x"] / df["delta_time"]
        # df["velocity_hit_y"] = df["delta_hit_y"] / df["delta_time"]
        df.loc[
            df.groupby("rally_id").head(1).index,
            [
                "velocity_A_x",
                "velocity_A_y",
                "velocity_B_x",
                "velocity_B_y",
                "velocity_C_x",
                "velocity_C_y",
                "velocity_D_x",
                "velocity_D_y",
                # "velocity_hit_x",
                # "velocity_hit_y",
            ],
        ] = 0
        velocity = df[
            [
                "velocity_A_x",
                "velocity_A_y",
                "velocity_B_x",
                "velocity_B_y",
                "velocity_C_x",
                "velocity_C_y",
                "velocity_D_x",
                "velocity_D_y",
                # "velocity_hit_x",
                # "velocity_hit_y",
            ]
        ].values

        # group difference
        grouped = df.groupby("rally_id")

        diff_df = grouped.apply(
            lambda x: x.assign(
                player_A_B_x_diff=x["player_A_x"] - x["player_B_x"],
                player_A_B_y_diff=x["player_A_y"] - x["player_B_y"],
                player_B_A_x_diff=x["player_B_x"] - x["player_A_x"],
                player_B_A_y_diff=x["player_B_y"] - x["player_A_y"],
                player_C_D_x_diff=x["player_C_x"] - x["player_D_x"],
                player_C_D_y_diff=x["player_C_y"] - x["player_D_y"],
                player_D_C_x_diff=x["player_D_x"] - x["player_C_x"],
                player_D_C_y_diff=x["player_D_y"] - x["player_C_y"],
            )
        ).reset_index(drop=True)

        for col in [
            "player_A_B_x_diff",
            "player_A_B_y_diff",
            "player_B_A_x_diff",
            "player_B_A_y_diff",
            "player_C_D_x_diff",
            "player_C_D_y_diff",
            "player_D_C_x_diff",
            "player_D_C_y_diff",
        ]:
            df[col] = diff_df[col]

        def calculate_distance(row):
            player_x = row[f'player_{row["player"]}_x']
            player_y = row[f'player_{row["player"]}_y']
            hit_x = row["hit_x"]
            hit_y = row["hit_y"]
            return abs(player_x - hit_x), abs(player_y - hit_y)

        distance = df.apply(calculate_distance, axis=1)

        df["hit_x_diff"] = distance.apply(lambda x: x[0])
        df["hit_y_diff"] = distance.apply(lambda x: x[1])

        group_differences = df[
            [
                "player_A_B_x_diff",
                "player_A_B_y_diff",
                "player_B_A_x_diff",
                "player_B_A_y_diff",
                "player_C_D_x_diff",
                "player_C_D_y_diff",
                "player_D_C_x_diff",
                "player_D_C_y_diff",
                # "hit_x_diff",
                # "hit_y_diff",
            ]
        ].values

        # shot_types
        shot_types = self.shot_type_enc.fit_transform(
            df[["ball_type"]].values
        ).toarray()

        # hit_players
        hit_player = self.hit_player_enc.fit_transform(df[["player"]].values).toarray()

        # direction

        def map_angle_to_direction(angle):
            adjusted_angle = (angle + 360) % 360
            direction = np.floor(adjusted_angle / 22.5)
            return direction
        df['angle_A'] = np.degrees(np.arctan2(df['delta_A_y'], df['delta_A_x']))
        df['direction_A'] = df['angle_A'].apply(map_angle_to_direction)
        
        df['angle_B'] = np.degrees(np.arctan2(df['delta_B_y'], df['delta_B_x']))
        df['direction_B'] = df['angle_B'].apply(map_angle_to_direction)

        df['angle_C'] = np.degrees(np.arctan2(df['delta_C_y'], df['delta_C_x']))
        df['direction_C'] = df['angle_C'].apply(map_angle_to_direction)

        df['angle_D'] = np.degrees(np.arctan2(df['delta_D_y'], df['delta_D_x']))
        df['direction_D'] = df['angle_D'].apply(map_angle_to_direction)

        df.loc[
            df.groupby("rally_id").head(1).index,
            [
                "direction_A",
                "direction_B",
                "direction_C",
                "direction_D",
            ],
        ] = 0

        direction = df[
            [
                "direction_A",
                "direction_A",
                "direction_B",
                "direction_B",
                "direction_C",
                "direction_C",
                "direction_D",
                "direction_D",
            ]
        ].values

        unique_rallies = np.unique(df["rally_id"])
        new_locs = []
        new_vels = []
        new_shot = []
        new_hit = []
        new_group = []
        new_direction = []
        for rally_id in unique_rallies:
            rally_data = df[df["rally_id"] == rally_id]
            n_timestamps = len(rally_data)
        
           

            indices = df["rally_id"] == rally_id

            locs_rally = displacements[indices].reshape(n_timestamps, self.n_players, 2)
            vels_rally = velocity[indices].reshape(n_timestamps, self.n_players, 2)
            group_rally = group_differences[indices].reshape(
                n_timestamps, self.n_players, 2
            )
            shot_rally = np.repeat(
                shot_types[indices][:, np.newaxis, :], self.n_players, axis=1
            )
            shot_rally = np.expand_dims(shot_rally, axis=0).reshape(
                n_timestamps, self.n_players, shot_rally.shape[2]
            )

            hit_rally = np.repeat(
                hit_player[indices][:, np.newaxis, :], self.n_players, axis=1
            )
            hit_rally = np.expand_dims(hit_rally, axis=0).reshape(
                n_timestamps, self.n_players, hit_rally.shape[2]
            )

            # hit_players_rally = hit_player[df["rally_id"] == rally_id]
            # shot_rally = np.zeros((n_timestamps, self.n_players, shot_types.shape[1]))
            # hit_rally = np.zeros((n_timestamps, self.n_players, 1))
            
            direction_rally = direction[indices].reshape(n_timestamps, self.n_players, 2)


            # for t in range(n_timestamps):
            #     hitting_player_idx = np.argmax(hit_players_rally[t])
                
            #     # shot_type_for_player = shot_types[np.where(df["rally_id"] == rally_id)[0][t]]
            #     # shot_rally[t, hitting_player_idx] = shot_type_for_player
            #     hit_rally[t, hitting_player_idx] = np.ones(1)

            # hit_rally = np.repeat(
            #     hit_player[indices][:, np.newaxis, :], self.n_players, axis=1
            # )
            # hit_rally = np.expand_dims(hit_rally, axis=0).reshape(
            #     n_timestamps, self.n_players, hit_rally.shape[2]
            # )
            new_locs.append(locs_rally)
            new_vels.append(vels_rally)
            new_group.append(group_rally)
            new_shot.append(shot_rally)
            new_hit.append(hit_rally)
            new_direction.append(direction_rally)

        displacements = new_locs
        velocity = new_vels
        group = new_group
        shot_types = new_shot
        hit_player = new_hit
        direction = new_direction

        max_len = max(len(x) for x in displacements)
        locs_pad = []
        for seq in displacements:
            pad_len = max_len - len(seq)
            batch_pad = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
            locs_pad.append(batch_pad)
        displacements = np.stack(locs_pad, axis=0)

        vels_pad = []
        for seq in velocity:
            pad_len = max_len - len(seq)
            batch_pad = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
            vels_pad.append(batch_pad)
        velocity = np.stack(vels_pad, axis=0)

        group_pad = []
        for seq in group:
            pad_len = max_len - len(seq)
            batch_pad = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
            group_pad.append(batch_pad)
        group = np.stack(group_pad, axis=0)

        shot_pad = []
        for seq in shot_types:
            pad_len = max_len - len(seq)
            batch_pad = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
            shot_pad.append(batch_pad)
        shot_types = np.stack(shot_pad, axis=0)

        hit_pad = []
        for seq in hit_player:
            pad_len = max_len - len(seq)
            batch_pad = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
            hit_pad.append(batch_pad)
        hit_player = np.stack(hit_pad, axis=0)

        direction_pad = []
        for seq in direction:
            pad_len = max_len - len(seq)
            batch_pad = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
            direction_pad.append(batch_pad)
        direction = np.stack(direction_pad, axis=0)
        # displacements = np.transpose(displacements, (1, 0, 2, 3))
        # velocity = np.transpose(velocity, (1, 0, 2, 3))
        # group = np.transpose(group, (1, 0, 2, 3))
        # shot_types = np.transpose(shot_types, (1, 0, 2, 3))
        # hit_player = np.transpose(hit_player, (1, 0, 2, 3))
        # direction = np.transpose(direction, (1, 0, 2, 3))
        # seq_len, _, _, features = displacements.shape

        # displacements = displacements.reshape((seq_len, -1, features))
        # velocity = velocity.reshape((seq_len, -1, features))
        # group = group.reshape((seq_len, -1, features))
        # shot_types = shot_types.reshape((seq_len, -1, shot_types.shape[-1]))
        # hit_player = hit_player.reshape((seq_len, -1, hit_player.shape[-1]))
        # direction = direction.reshape((seq_len, -1, direction.shape[-1]))

        return displacements, velocity, group, shot_types, hit_player, direction

    def _save(
        self,
        locs: np.ndarray,
        vels: np.array,
        group: np.array,
        shot_types: np.ndarray,
        hit_players: np.ndarray,
        direction: np.ndarray,
        output_dir: Path,
    ) -> None:
        np.save(output_dir.joinpath("displacements.npy"), locs)
        np.save(output_dir.joinpath("velocity.npy"), vels)
        np.save(output_dir.joinpath("group.npy"), group)
        np.save(output_dir.joinpath("goals.npy"), shot_types)
        np.save(output_dir.joinpath("hit.npy"), hit_players)
        np.save(output_dir.joinpath("direction.npy"), direction)

    def preprocess(self):
        # Train
        train_df = pd.read_csv(self.train_path)
        train_output_dir = Path(self.args.root).joinpath("train")
        train_output_dir.mkdir(parents=True, exist_ok=True)
        locs, vels, group, shot_types, hit_players, direction = self._preprocess_single(train_df)

        self._save(
            locs=locs,
            vels=vels,
            group=group,
            shot_types=shot_types,
            hit_players=hit_players,
            direction=direction,
            output_dir=train_output_dir,
        )

        # Test
        test_df = pd.read_csv(self.test_path)
        test_output_dir = Path(self.args.root).joinpath("test")
        test_output_dir.mkdir(parents=True, exist_ok=True)
        locs, vels, group, shot_types, hit_players, direction = self._preprocess_single(test_df)

        self._save(
            locs=locs,
            vels=vels,
            group=group,
            shot_types=shot_types,
            hit_players=hit_players,
            direction=direction,
            output_dir=test_output_dir,
        )


def preprocess_cli(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("-n", "--n_agents", type=int, default=4, help="number of agents")  # fmt: skip
    parser.add_argument("-val", "--val_size", type=float, default=0.2, help="percentage of validation set")  # fmt: skip
    parser.add_argument("-r", "--root", type=str, default=join("data", "badminton", "doubles"), help="dataset directory")  # fmt: skip
    return parser


def preprocess_main(args: Namespace):
    preprocessor = BadmintonPreprocessor(args)
    preprocessor.preprocess()
