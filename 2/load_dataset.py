import argparse
import os

import lerobot
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_id", type=str, default="lerobot/pusht")
    # parser.add_argument("--dataset_id", type=str, default="lerobot/aloha_sim_insertion_human")
    parser.add_argument("--dataset_id", type=str, default="lerobot/aloha_static_coffee")
    parser.add_argument("--episodes", type=list, default=[0, 10, 11, 23])
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print("List of available datasets:", lerobot.available_datasets)

    # LeRobot のデータセットを読み込む
    dataset = LeRobotDataset(args.dataset_id, episodes=args.episodes)

    # データセットの概要を print する
    print("dataset:", dataset)
    # 以下のようなデータセットの概要内容が得られる
    # dataset: LeRobotDataset({
    #     Repository ID: 'lerobot/aloha_static_coffee',|█████████████████████████████████████████████| 6.84M/6.84M [00:00<00:00, 12.7MB/s]
    #     Number of selected episodes: '50',
    #     Number of selected samples: '55000',(…): 100%|█████████████████████████████████████████████| 6.51M/6.51M [00:00<00:00, 13.7MB/s]
    #     Features: '
    #         [
    #             'observation.images.cam_high',
    #             'observation.images.cam_left_wrist',
    #             'observation.images.cam_low',
    #             'observation.images.cam_right_wrist',
    #             'observation.state',
    #             'observation.effort',
    #             'action',
    #             'episode_index',
    #             'frame_index',
    #             'timestamp',
    #             'next.done',
    #             'index',
    #             'task_index'
    #         ]
    #     }
    # )

    # 特定のエピソードと時間ステップのデータを print する
    print("dataset[0]:", dataset[0])
    # 以下のようなデータセットの内容が得られる
    # {
    #   # 観測データ（observation）：ロボットの上部から撮影したカメラの画像
    #   'observation.images.cam_high': tensor([[[0.9216, 0.8980, 0.8902,  ..., 0.6000, 0.9020, 0.8941],
    #          [0.9059, 0.9176, 0.9098,  ..., 0.5686, 0.8784, 0.8941],
    #          [0.8392, 0.9020, 0.9294,  ..., 0.5216, 0.8471, 0.9059],
    #          ...,
    #          [0.8196, 0.8157, 0.8157,  ..., 0.5020, 0.5020, 0.5020]]]),
    #   # 観測データ（observation）：ロボットの左手首に取り付けられたカメラの画像
    #   'observation.images.cam_left_wrist': tensor([[[0.1569, 0.1608, 0.1608,  ..., 0.2863, 0.3255, 0.3490],
    #          [0.1647, 0.1608, 0.1529,  ..., 0.2863, 0.3255, 0.3490],
    #          [0.1804, 0.1569, 0.1373,  ..., 0.2863, 0.3216, 0.3451],
    #          ...,
    #          [0.2549, 0.2549, 0.2549,  ..., 0.1176, 0.1176, 0.1176]]]),
    #   # 観測データ（observation）：ロボットの下部から撮影したカメラ画像
    #   'observation.images.cam_low': tensor([[[0.9176, 0.9176, 0.9176,  ..., 0.4235, 0.4235, 0.4196],
    #          [0.9176, 0.9176, 0.9176,  ..., 0.4235, 0.4235, 0.4196],
    #          [0.9176, 0.9176, 0.9176,  ..., 0.4235, 0.4196, 0.4196],
    #          ...,
    #          [0.9098, 0.9098, 0.9098,  ..., 0.5451, 0.5451, 0.5451]]]),
    #   # 観測データ（observation）：ロボットの右手首に取り付けられたカメラの画像
    #   'observation.images.cam_right_wrist': tensor([[[0.4745, 0.4706, 0.4706,  ..., 0.6627, 0.6627, 0.6627],
    #          [0.4706, 0.4667, 0.4627,  ..., 0.6588, 0.6588, 0.6588],
    #          [0.4667, 0.4667, 0.4627,  ..., 0.6588, 0.6588, 0.6588],
    #          ...,
    #          [0.0980, 0.0941, 0.0941,  ..., 0.1059, 0.1059, 0.1059]]]),
    #   # 観測データ（observation）：状態（state）ベクトル。関節角度や位置情報など
    #   'observation.state': tensor([-0.0031, -0.9664,  1.1888, -0.0015, -0.2915,  0.0015,  0.0130,  0.0031,
    #         -0.9695,  1.1873, -0.0015, -0.2899,  0.0015,  0.0044]),
    #   # 観測データ（observation）：ロボットの各関節にかかる力やトルク情報
    #   'observation.effort': tensor([   0.0000,  139.8800, -731.6800,    0.0000, -228.6500,   -5.3800,
    #         -527.2400,  -64.5600,  220.5800, -739.7500,    0.0000, -269.0000,
    #           -2.6900, -217.8900]),
    #   # 行動（action）ベクトル：ロボットが次に実行するアクション（関節の目標位置や速度など）
    #   'action': tensor([-0.0123, -0.9557,  1.1428, -0.0031, -0.2930, -0.0123,  0.1778,  0.0061,
    #         -0.9541,  1.1551, -0.0015, -0.3037,  0.0031,  0.0588]),
    #   # エピソードの値
    #   'episode_index': tensor(0),
    #   # フレーム（時間ステップ？）
    #   'frame_index': tensor(0),
    #   # 時間情報
    #   'timestamp': tensor(0.),
    #   # エピソードが終了したかどうかのフラグ
    #   'next.done': tensor(False),
    #   # データセット内の絶対的なインデックス
    #   'index': tensor(0),
    #   # タスクの種類を示すインデックス
    #   'task_index': tensor(0),
    #   # 制御指示テキスト
    #   'task': "Place the coffee capsule inside the capsule container,
    #           then place the cup onto the center of the cup tray,
    #           then push the 'Hot Water' and 'Travel Mug' buttons."
    # }
