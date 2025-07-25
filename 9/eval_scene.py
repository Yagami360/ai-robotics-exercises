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
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.fourier import GR1T2_CFG  # isort: skip


class GR1SceneCfg(InteractiveSceneCfg):
    # 地面を配置
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # 照明を配置
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # テーブルを配置
    counter = AssetBaseCfg(
        prim_path="/World/Counter",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.3, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 固定オブジェクト
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.27, 0.07),
                roughness=0.7,
                metallic=0.0,
                opacity=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 1.15),
        ),
    )

    # 梨を配置
    pear = AssetBaseCfg(
        prim_path="/World/Pear",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,  # 動かせるオブジェクト
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.65, 0.45, 0.18),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.35, -0.07, 1.22),
        ),
    )

    # プレート（皿）を配置
    plate = AssetBaseCfg(
        prim_path="/World/Plate",
        spawn=sim_utils.CylinderCfg(
            radius=0.06,
            height=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 動かせるオブジェクト
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.25, 0.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.50, -0.05, 1.22),
        ),
    )

    # センサー：ロボット頭部にカメラを追加
    sensor_camera = CameraCfg(
        prim_path="/World/Robot/head_link/Camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=300.0,
            horizontal_aperture=40.0,
            vertical_aperture=40.0,
            clipping_range=(0.05, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.35, 0.0, 0.50),
            # オイラー角 X,Y,Z = (180.0, 0.0, -73.0) 相当のクオータニオン (w,x,y,z) になる
            rot=(0.0, 1.0, 0.0, 0.0),
            # rot=(1.0, 0.0, 0.0, 0.0),   # NG
            # rot=(0.0, 0.0, 0.0, 1.0),
            # TODO: オイラー角 X,Y,Z = (0.0, 0.0, -90.0) 相当のクオータニオン (w,x,y,z) になるようにする
            # rot=(xxx, xxx, xxx, xxx)
        ),
        # NOTE: Model was trained with rgb only
        data_types=["rgb"],
        # data_types=["rgb", "depth"],
        height=256,
        width=256,
    )

    # GR-1-T2 ロボットを配置
    robot: ArticulationCfg = GR1T2_CFG.replace(
        prim_path="/World/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )


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
sim_cfg = sim_utils.SimulationCfg(device=args.device)
sim = sim_utils.SimulationContext(sim_cfg)

# シーンを作成
scene_cfg = GR1SceneCfg(num_envs=args.num_envs, env_spacing=2.0)
scene = InteractiveScene(scene_cfg)
robot = scene["robot"]
sensor_camera = scene["sensor_camera"]

print(f"シーン作成完了: {scene}")
# print(f"robot: {vars(robot)}")
# print(f"sensor_camera: {vars(sensor_camera)}")

# カメラを配置
sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

# シミュレーションをリセット
sim.reset()

# GR-1ロボットのジョイント設定
print(f"ジョイント数: {len(robot.joint_names)}")
print("ジョイント一覧:\n" + ",\n".join(sorted(robot.joint_names)))

left_arm_joint_names = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
]

right_arm_joint_names = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]

left_hand_joint_names = [
    # 小指
    "L_pinky_proximal_joint",
    # 薬指
    "L_ring_proximal_joint",
    # 中指
    "L_middle_proximal_joint",
    # 人差し指
    "L_index_proximal_joint",
    # 親指の回転
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    # GR-1-T2 ロボットの指関節点の次元数は11次元であるが、学習済みモデル（＆データセット）の指関節点の次元数は6次元になるので、以下の関節点は使用しない
    # "L_thumb_distal_joint",
    # "L_index_intermediate_joint",
    # "L_middle_intermediate_joint",
    # "L_pinky_intermediate_joint",
    # "L_ring_intermediate_joint",
]

right_hand_joint_names = [
    # 小指
    "R_pinky_proximal_joint",
    # 薬指
    "R_ring_proximal_joint",
    # 中指
    "R_middle_proximal_joint",
    # 人差し指
    "R_index_proximal_joint",
    # 親指の回転
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    # GR-1-T2 ロボットの指関節点の次元数は11次元であるが、学習済みモデル（＆データセット）の指関節点の次元数は6次元になるので、以下の関節点は使用しない
    # "R_thumb_distal_joint",
    # "R_index_intermediate_joint",
    # "R_middle_intermediate_joint",
    # "R_pinky_intermediate_joint",
    # "R_ring_intermediate_joint",
]

if config_key == "gr1_arms_waist":
    waist_joint_names = [
        "waist_yaw_joint",
        "waist_pitch_joint",
        "waist_roll_joint",
    ]

left_arm_joint_ids = [
    robot.joint_names.index(name)
    for name in left_arm_joint_names
    if name in robot.joint_names
]
right_arm_joint_ids = [
    robot.joint_names.index(name)
    for name in right_arm_joint_names
    if name in robot.joint_names
]

left_hand_joint_ids = [
    robot.joint_names.index(name)
    for name in left_hand_joint_names
    if name in robot.joint_names
]
right_hand_joint_ids = [
    robot.joint_names.index(name)
    for name in right_hand_joint_names
    if name in robot.joint_names
]

if config_key == "gr1_arms_waist":
    waist_joint_ids = [
        robot.joint_names.index(name)
        for name in waist_joint_names
        if name in robot.joint_names
    ]

print(f"左腕ジョイント: {left_arm_joint_names}")
print(f"右腕ジョイント: {right_arm_joint_names}")
print(f"左腕ジョイントID: {left_arm_joint_ids}")
print(f"右腕ジョイントID: {right_arm_joint_ids}")
print(f"左手ジョイント: {left_hand_joint_names}")
print(f"右手ジョイント: {right_hand_joint_names}")
print(f"左手ジョイントID: {left_hand_joint_ids}")
print(f"右手ジョイントID: {right_hand_joint_ids}")
if config_key == "gr1_arms_waist":
    print(f"腰部ジョイント: {waist_joint_names}")
    print(f"腰部ジョイントID: {waist_joint_ids}")

# シミュレーション実行
sim_dt = sim.get_physics_dt()
sim_time = 0.0
count = 0
action_step = 0

print("シミュレーション開始...")

action_buffer = {
    "left_arm": [],
    "right_arm": [],
    "left_hand": [],
    "right_hand": [],
}
if config_key == "gr1_arms_waist":
    action_buffer["waist"] = []

while simulation_app.is_running():
    # バッファが空なら推論を実行
    if len(action_buffer["left_arm"]) == 0:
        # ------------------------------------------------------------
        # 観測値（入力データ）の更新
        # ------------------------------------------------------------
        # ロボットの各関節のジョイント取得
        joint_pos = robot.data.joint_pos[0].cpu().numpy().astype(np.float32)

        # ロボットの各関節のジョイント速度
        # NOTE: 本学習済みモデルでは使用しない
        # joint_vel = robot.data.joint_vel[0].cpu().numpy().astype(np.float32)

        # 腕のジョイント位置
        left_arm_state = np.zeros((1, 7), dtype=np.float32)
        if len(left_arm_joint_ids) >= 7:
            left_arm_state[0] = joint_pos[left_arm_joint_ids[:7]]
        elif len(left_arm_joint_ids) > 0:
            left_arm_state[0, : len(left_arm_joint_ids)] = joint_pos[left_arm_joint_ids]

        right_arm_state = np.zeros((1, 7), dtype=np.float32)
        if len(right_arm_joint_ids) >= 7:
            right_arm_state[0] = joint_pos[right_arm_joint_ids[:7]]
        elif len(right_arm_joint_ids) > 0:
            right_arm_state[0, : len(right_arm_joint_ids)] = joint_pos[
                right_arm_joint_ids
            ]

        # 手のジョイント位置
        left_hand_state = np.zeros((1, 6), dtype=np.float32)
        right_hand_state = np.zeros((1, 6), dtype=np.float32)
        if len(left_hand_joint_ids) >= 6:
            left_hand_state[0] = joint_pos[left_hand_joint_ids[:6]]
        elif len(left_hand_joint_ids) > 0:
            left_hand_state[0, : len(left_hand_joint_ids)] = joint_pos[
                left_hand_joint_ids
            ]
        if len(right_hand_joint_ids) >= 6:
            right_hand_state[0] = joint_pos[right_hand_joint_ids[:6]]
        elif len(right_hand_joint_ids) > 0:
            right_hand_state[0, : len(right_hand_joint_ids)] = joint_pos[
                right_hand_joint_ids
            ]

        # 腰部のジョイント位置
        if config_key == "gr1_arms_waist":
            waist_state = np.zeros((1, 3), dtype=np.float32)
            if len(waist_joint_ids) >= 3:
                waist_state[0] = joint_pos[waist_joint_ids[:3]]
            elif len(waist_joint_ids) > 0:
                waist_state[0, : len(waist_joint_ids)] = joint_pos[waist_joint_ids]

        # ロボットのカメラからの画像データ
        camera_image = sensor_camera.data.output["rgb"].cpu().numpy()
        cv2.imwrite(
            "robot_camera.png", cv2.cvtColor(camera_image[0], cv2.COLOR_RGB2BGR)
        )

        # 推論用の入力データを作成
        observation = {
            "state.left_arm": left_arm_state,
            "state.right_arm": right_arm_state,
            "state.left_hand": left_hand_state,
            "state.right_hand": right_hand_state,
            "video.ego_view": camera_image,
            "task_description": [
                "pick the pear from the counter and place it in the plate",
            ],
        }
        if config_key == "gr1_arms_waist":
            observation["state.waist"] = waist_state

        if count == 0:
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
            if count == 0:
                for k in action_chunk.keys():
                    if isinstance(action_chunk[k], list):
                        print(f"action_chunk[{k}]: {action_chunk[k]}")
                    else:
                        print(
                            f"action_chunk[{k}] shape: {action_chunk[k].shape}, dtype: {action_chunk[k].dtype}, min: {action_chunk[k].min()}, max: {action_chunk[k].max()}, mean: {action_chunk[k].mean()}"
                        )

        # 各アクションをバッファに格納（list化してpop(0)で取り出せるように）
        for k in action_buffer.keys():
            if args.denorm_action:
                modality_key = k
                norm_actions = np.array(action_chunk[f"action.{k}"])
                denorm_actions = np.array([denormalize_action(modality_key, a) for a in norm_actions])
                action_buffer[k] = list(denorm_actions)
            else:
                action_buffer[k] = list(action_chunk[f"action.{k}"])

    # バッファから1ステップ分のアクションを取り出す
    left_arm_action = torch.tensor(
        action_buffer["left_arm"].pop(0), device=args.device, dtype=torch.float32
    )
    if count == 0:
        print(
            f"left_arm_action shape: {left_arm_action.shape}, dtype: {left_arm_action.dtype}, min: {left_arm_action.min()}, max: {left_arm_action.max()}"
        )

    right_arm_action = torch.tensor(
        action_buffer["right_arm"].pop(0), device=args.device, dtype=torch.float32
    )
    if count == 0:
        print(
            f"right_arm_action shape: {right_arm_action.shape}, dtype: {right_arm_action.dtype}, min: {right_arm_action.min()}, max: {right_arm_action.max()}"
        )

    left_hand_action = torch.tensor(
        action_buffer["left_hand"].pop(0), device=args.device, dtype=torch.float32
    )
    if count == 0:
        print(
            f"left_hand_action shape: {left_hand_action.shape}, dtype: {left_hand_action.dtype}, min: {left_hand_action.min()}, max: {left_hand_action.max()}"
        )

    right_hand_action = torch.tensor(
        action_buffer["right_hand"].pop(0), device=args.device, dtype=torch.float32
    )
    if count == 0:
        print(
            f"right_hand_action shape: {right_hand_action.shape}, dtype: {right_hand_action.dtype}, min: {right_hand_action.min()}, max: {right_hand_action.max()}"
        )

    if config_key == "gr1_arms_waist":
        waist_action = torch.tensor(
            action_buffer["waist"].pop(0), device=args.device, dtype=torch.float32
        )
        if count == 0:
            print(
                f"waist_action shape: {waist_action.shape}, dtype: {waist_action.dtype}, min: {waist_action.min()}, max: {waist_action.max()}"
            )

    # ロボットにアクションを適用
    robot.set_joint_position_target(left_arm_action, joint_ids=left_arm_joint_ids)
    robot.set_joint_position_target(right_arm_action, joint_ids=right_arm_joint_ids)
    robot.set_joint_position_target(
        left_hand_action,
        # 学習済み次元数が 6 次元で GR-1-T2 の次元数が 11 次元になるので、対応付けを行う
        joint_ids=left_hand_joint_ids[: len(left_hand_action)],
    )
    robot.set_joint_position_target(
        right_hand_action,
        # 学習済み次元数が 6 次元で GR-1-T2 の次元数が 11 次元になるので、対応付けを行う
        joint_ids=right_hand_joint_ids[: len(right_hand_action)],
    )
    if config_key == "gr1_arms_waist":
        robot.set_joint_position_target(
            waist_action, joint_ids=waist_joint_ids[: len(waist_action)]
        )

    # シミュレーションステップ実行
    scene.write_data_to_sim()
    sim.step()
    sim_time += sim_dt
    count += 1
    scene.update(sim_dt)

# シミュレーション終了
simulation_app.close()
