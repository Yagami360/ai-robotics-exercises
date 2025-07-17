import argparse
import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="GR-1 Robot Simulation with Isaac-GR00T")
parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1-2B")
# parser.add_argument("--model_path", type=str, default="../checkpoints/gr00t/checkpoint-1000/")
parser.add_argument("--dataset_path", type=str, default="../Isaac-GR00T/demo_data/robot_sim.PickNPlace")
parser.add_argument("--config_key", type=str, default="gr1_arms_only", choices=["gr1_arms_only", "gr1_arms_waist"])
parser.add_argument("--denorm_action", type=bool, default=False)
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
# シーン定義
# NOTE: "ModuleNotFoundError: No module named 'isaacsim.core'" のエラーがでないように、
# IsaacSim 関連の import 文は AppLauncher の後に記載する必要がある
# ------------------------------------------------------------
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env import FrankaCabinetEnvCfg
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
sensor_camera = CameraCfg(
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
# Isaac-GR00T モデル定義
# ------------------------------------------------------------
import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

# 利用可能なモデルを確認
print(f"利用可能なモデル: {list(DATA_CONFIG_MAP.keys())}")
config_key = args.config_key
data_config = DATA_CONFIG_MAP[config_key]
print(f"data_config.modality_config(): {data_config.modality_config()}")

policy = Gr00tPolicy(
    model_path=args.model_path,
    modality_config=data_config.modality_config(),
    modality_transform=data_config.transform(),
    embodiment_tag=(
        EmbodimentTag.GR1
        if args.model_path == "nvidia/GR00T-N1-2B"
        else EmbodimentTag.NEW_EMBODIMENT
    ),
    device=args.device,
)

def policy_to_dict(obj):
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ["min", "max", "mean", "std", "q01", "q99"] and isinstance(v, str):
                arr = np.fromstring(v.replace("\n", " "), sep=" ").tolist()
                result[k] = arr
            else:
                result[k] = policy_to_dict(v)
        return result
    elif isinstance(obj, (list, tuple)):
        return [policy_to_dict(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return {k: policy_to_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)

policy_dict = policy_to_dict(policy)
with open("policy.json", "w", encoding="utf-8") as f:
    json.dump(policy_dict, f, ensure_ascii=False, indent=4)

# 出力 action の逆正規化処理のためのメタデータ読み取り
# meta_stats = policy_dict["metadata"]["statistics"]
# meta_modality = policy_dict["metadata"]["modalities"]
# print(f"meta_stats: {meta_stats}")
# print(f"meta_modality: {meta_modality}")

with open(os.path.join(args.dataset_path, "meta", "stats.json")) as f:
    meta_stats = json.load(f)
with open(os.path.join(args.dataset_path, "meta", "modality.json")) as f:
    meta_modality = json.load(f)
print(f"meta_stats: {meta_stats}")
print(f"meta_modality: {meta_modality}")

def denormalize_action(modality_key, normalized_action):
    start = meta_modality["action"][modality_key]["start"]
    end = meta_modality["action"][modality_key]["end"]
    min_ = np.array(meta_stats["action"]["min"][start:end])
    max_ = np.array(meta_stats["action"]["max"][start:end])
    norm = normalized_action
    if isinstance(norm, torch.Tensor):
        norm = norm.cpu().numpy()
    norm = norm[: end - start]
    min_ = min_[: end - start]
    max_ = max_[: end - start]
    return norm * (max_ - min_) + min_

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

