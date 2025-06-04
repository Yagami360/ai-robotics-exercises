import argparse
import os

import cv2
import numpy as np
import torch

parser = argparse.ArgumentParser(description="GR-1 Robot Simulation with Isaac-GR00T")
parser.add_argument(
    "--model_path", type=str, default="nvidia/GR00T-N1-2B", help="GR00T model path"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)

# ------------------------------------------------------------
# シミュレーターアプリ作成
# ------------------------------------------------------------
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# シード設定
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ------------------------------------------------------------
# Isaac-GR00T モデル定義
# ------------------------------------------------------------
import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

# 利用可能なモデルを確認
print(f"利用可能なモデル: {list(DATA_CONFIG_MAP.keys())}")
config_key = "gr1_arms_waist"
data_config = DATA_CONFIG_MAP[config_key]

policy = Gr00tPolicy(
    model_path=args.model_path,
    modality_config=data_config.modality_config(),
    modality_transform=data_config.transform(),
    embodiment_tag=EmbodimentTag.GR1,
    device=args.device,
)
# ------------------------------------------------------------
# シーン & 環境定義
# NOTE: "ModuleNotFoundError: No module named 'isaacsim.core'" のエラーがでないように、
# IsaacSim 関連の import 文は AppLauncher の後に記載する必要がある
# ------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import (
    PickPlaceGR1T2EnvCfg
)

print(f"ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")

# 環境の設定
env_cfg = PickPlaceGR1T2EnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = args.device
if args.device == "cpu":
    env_cfg.sim.use_fabric = False

# 環境の作成
env = ManagerBasedRLEnv(cfg=env_cfg)
print(f"環境作成完了: {env}")
print(f"観測空間: {env.observation_manager}")
print(f"アクション空間: {env.action_manager}")

# ロボットのカメラ追加
# env.scene.sensor_camera = CameraCfg(
#     prim_path="/World/Robot/head_link/Camera",
#     spawn=sim_utils.PinholeCameraCfg(
#         focal_length=16.0,
#         focus_distance=300.0,
#         horizontal_aperture=40.0,
#         vertical_aperture=40.0,
#         clipping_range=(0.05, 1.0e5),
#     ),
#     offset=CameraCfg.OffsetCfg(
#         pos=(0.1, 0.0, 0.7),
#         rot=(0.0, 1.0, 0.0, 0.0),
#     ),
#     data_types=["rgb", "depth"],
#     height=256,
#     width=256,
# )

# ロボットと関連する情報を取得
robot = env.scene["robot"]
if "object" in env.scene:
    object_entity = env.scene["object"]
    print(f"オブジェクト: {object_entity}")

print(f"ロボット: {robot}")
print(f"ロボットのジョイント一覧: {robot.joint_names}")

# ------------------------------------------------------------
# シミュレーション実行
# ------------------------------------------------------------
# 環境をリセット
observations, _ = env.reset()
print(f"初期観測: {observations}")

# シミュレーション実行
sim_dt = env.physics_dt
sim_time = 0.0
count = 0

print("シミュレーション開始...")

while simulation_app.is_running():
    # 一定ステップごとにGR00Tで推論を実行
    if count % 100 == 0:
        print(f"シミュレーション時間: {sim_time:.2f}秒")

        # ------------------------------------------------------------
        # 入力データ（observation）を設定
        # ------------------------------------------------------------
        # ロボットの各関節のジョイント位置
        joint_pos = robot.data.joint_pos[0].cpu().numpy().astype(np.float32)

        # ロボットのカメラからの画像データ
        # camera_data = sensor_camera.data
        # rgb_image = camera_data.output["rgb"][0].cpu().numpy()
        # camera_image = (rgb_image * 255).astype(np.uint8)
        # camera_image = camera_image.reshape(1, 256, 256, 3)
        # image_to_save = camera_image[0]
        # cv2.imwrite("robot_camera.png", cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))

        # 推論用の入力データを作成
        observation = {
            "state.left_arm": joint_pos[
                robot.joint_names.index(
                    "left_shoulder_pitch_joint"
                ) : robot.joint_names.index("left_wrist_pitch_joint")
                + 1
            ].reshape(1, -1),
            "state.right_arm": joint_pos[
                robot.joint_names.index(
                    "right_shoulder_pitch_joint"
                ) : robot.joint_names.index("right_wrist_pitch_joint")
                + 1
            ].reshape(1, -1),
            "state.left_hand": np.zeros((1, 6), dtype=np.float32),
            "state.right_hand": np.zeros((1, 6), dtype=np.float32),
            "state.waist": joint_pos[
                robot.joint_names.index("waist_yaw_joint") : robot.joint_names.index(
                    "waist_roll_joint"
                )
                + 1
            ].reshape(1, -1),
            "video.ego_view": np.zeros((1, 256, 256, 3), dtype=np.uint8),
            "task_description": ["pick the cylinder object and place it on the table"],
        }

        # GR00Tモデルで推論
        with torch.inference_mode():
            action_chunk = policy.get_action(observation)
            print(f"action_chunk keys: {action_chunk.keys()}")

        # アクションを環境のフォーマットに変換
        # TODO: action_chunkをPinkIKアクションフォーマットに変換する処理を実装
        actions = env_cfg.idle_action.unsqueeze(0).to(env.device)

    # 環境のステップ実行
    observations, rewards, dones, truncated, info = env.step(actions)

    sim_time += sim_dt
    count += 1

# 環境終了
env.close()

# シミュレーション終了
simulation_app.close()
