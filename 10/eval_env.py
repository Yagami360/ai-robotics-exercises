import argparse
import os

import cv2
import numpy as np
import torch

# コマンドライン引数の設定
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="../checkpoints/gr00t.single_panda_gripper.OpenDrawer/checkpoint-3000",
)
parser.add_argument(
    "--data_config",
    type=str,
    default="single_panda_gripper",
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
data_config = DATA_CONFIG_MAP[args.data_config]

policy = Gr00tPolicy(
    model_path=args.model_path,
    modality_config=data_config.modality_config(),
    modality_transform=data_config.transform(),
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    device=args.device,
)

# ------------------------------------------------------------
# 環境定義
# NOTE: "ModuleNotFoundError: No module named 'isaacsim.core'" のエラーがでないように、
# IsaacSim 関連の import 文は AppLauncher の後に記載する必要がある
# ------------------------------------------------------------
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env import (
    FrankaCabinetEnvCfg
)
from isaaclab_tasks.utils import load_cfg_from_registry

env_cfg = FrankaCabinetEnvCfg()
# env_cfg = load_cfg_from_registry(
#     "Isaac-Reach-Franka-v0",
#     kwargs={
#         "env_cfg_entry_point": "env_cfg_entry_point",
#         "robomimic_bc_cfg_entry_point": "robomimic_bc_cfg_entry_point",
#     },
# )
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = args.device
if args.device == "cpu":
    env_cfg.sim.use_fabric = False

print("env_cfg structure:")
print(f"env_cfg type: {type(env_cfg)}")
print(f"env_cfg.scene type: {type(env_cfg.scene)}")
print(f"env_cfg.scene attributes: {dir(env_cfg.scene)}")
print(f"env_cfg.scene vars: {vars(env_cfg.scene)}")

# カメラを環境に追加
wrist_camera = CameraCfg(
    prim_path="/World/Camera",
    # prim_path="{ENV_REGEX_NS}/Robot/panda_hand/Camera",
    # prim_path="{ENV_REGEX_NS}/Camera",
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.0,
        focus_distance=300.0,
        horizontal_aperture=40.0,
        vertical_aperture=40.0,
        clipping_range=(0.05, 1.0e5),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(1.0, 0.0, 0.25),
        rot=(0.0, 0.0, 0.0, -1.0),
    ),
    # NOTE: Model was trained with rgb only
    data_types=["rgb"],
    # data_types=["rgb", "depth"],
    height=256,
    width=256,
)

env_cfg.scene.wrist_camera = wrist_camera

# 環境の作成
env = gym.make("Isaac-Franka-Cabinet-Direct-v0", cfg=env_cfg)
# env = ManagerBasedRLEnv(cfg=env_cfg)

print(f"env.observation_space: {env.observation_space}")
# env.observation_space: Box(-inf, inf, (1, 23), float32)

print(f"env.action_space: {env.action_space}")
# env.action_space: Box(-inf, inf, (1, 9), float32)

print(f"env: {vars(env)}")

print(f"env.unwrapped._robot.data.joint_names: {env.unwrapped._robot.data.joint_names}")

# ------------------------------------------------------------
# シミュレーション実行
# ------------------------------------------------------------
# 環境をリセット
observation_env, info = env.reset()
for k in observation_env.keys():
    if isinstance(observation_env[k], np.ndarray) or isinstance(
        observation_env[k], torch.Tensor
    ):
        print(
            f"observation_env[{k}] shape: {observation_env[k].shape}, dtype: {observation_env[k].dtype}, min: {observation_env[k].min()}, max: {observation_env[k].max()}"
        )
    else:
        print(f"observation_env[{k}]: {observation_env[k]}")
# observation_env[policy] shape: torch.Size([1, 23]), dtype: torch.float32, min: -0.6742031574249268, max: 0.9862452745437622

print(f"info: {info}")

# シミュレーション実行
step = 0

print("シミュレーション開始...")

while simulation_app.is_running():
    # ------------------------------------------------------------
    # 観測値（入力データ）の更新
    # ------------------------------------------------------------
    # ロボットの各関節のジョイント取得
    state = observation_env["policy"][0].cpu().numpy()
    print(f"state.shape: {state.shape}")
    print(f"state: {state}")

    joint_positions = state[0:9]  # 0〜8まで
    gripper_qpos = state[7:9]  # panda_finger_joint1, panda_finger_joint2
    joint_velocities = state[9:16]
    to_target = state[16:19]
    drawer_pos = state[19]
    drawer_vel = state[20]

    # ロボットのカメラからの画像データ
    wrist_camera_image = (
        env.unwrapped.scene.sensors["wrist_camera"].data.output["rgb"].cpu().numpy()
    )
    cv2.imwrite(
        "env_wrist_camera.png", cv2.cvtColor(wrist_camera_image[0], cv2.COLOR_RGB2BGR)
    )

    right_camera_image = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    cv2.imwrite(
        "env_right_camera.png", cv2.cvtColor(right_camera_image[0], cv2.COLOR_RGB2BGR)
    )

    left_camera_image = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    cv2.imwrite(
        "env_left_camera.png", cv2.cvtColor(left_camera_image[0], cv2.COLOR_RGB2BGR)
    )

    # 推論用の入力データを作成
    observation = {
        "state.end_effector_position_relative": to_target.reshape(1, -1),
        "state.end_effector_rotation_relative": np.zeros(
            (1, 4), dtype=np.float32
        ),  # 回転情報は利用できないため0で埋める
        "state.gripper_qpos": gripper_qpos.reshape(1, -1),
        "state.base_position": np.zeros(
            (1, 3), dtype=np.float32
        ),  # ベース位置は利用できないため0で埋める
        "state.base_rotation": np.array(
            [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),  # ベース回転は利用できないため単位クォータニオンで埋める
        "video.wrist_view": wrist_camera_image.astype(np.uint8),
        "video.right_view": right_camera_image.astype(np.uint8),
        "video.left_view": left_camera_image.astype(np.uint8),
        "task_description": ["OpenDrawer"],
    }

    for k in observation.keys():
        if isinstance(observation[k], list):
            print(f"observation[{k}]: {observation[k]}")
        else:
            print(
                f"observation[{k}] shape: {observation[k].shape}, dtype: {observation[k].dtype}, min: {observation[k].min()}, max: {observation[k].max()}"
            )

    # Isaac-GR00T モデルの推論処理
    with torch.inference_mode():
        action_chunk = policy.get_action(observation)
        for k in action_chunk.keys():
            if isinstance(action_chunk[k], list):
                print(f"action_chunk[{k}]: {action_chunk[k]}")
            else:
                print(
                    f"action_chunk[{k}] shape: {action_chunk[k].shape}, dtype: {action_chunk[k].dtype}, min: {action_chunk[k].min()}, max: {action_chunk[k].max()}"
                )

    # 行動ベクトルを取得
    action = torch.zeros((1, 9), dtype=torch.float32, device=args.device)
    action[0, 0:3] = torch.from_numpy(
        action_chunk["action.end_effector_position"][0]
    ).to(args.device)
    action[0, 3:6] = torch.from_numpy(
        action_chunk["action.end_effector_rotation"][0]
    ).to(args.device)
    action[0, 6] = torch.tensor(
        action_chunk["action.gripper_close"][0], dtype=torch.float32, device=args.device
    )
    action[0, 7:9] = torch.from_numpy(action_chunk["action.base_motion"][0, :2]).to(
        args.device
    )

    # シミュレーションステップ実行
    observation_env, rewards, dones, truncated, info = env.step(action)
    print(f"{step=} {rewards=} {dones=} {truncated=}")
    # env.scene.wrist_camera = wrist_camera
    step += 1

# 環境終了
env.close()

# シミュレーション終了
simulation_app.close()
