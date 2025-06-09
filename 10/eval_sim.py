import argparse
import os

import cv2
import numpy as np
import torch

# コマンドライン引数の設定
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1-2B")
parser.add_argument(
    "--data_config",
    type=str,
    choices=["bimanual_panda_gripper", "bimanual_panda_hand"],
    default="bimanual_panda_gripper",
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
# シーン定義
# NOTE: "ModuleNotFoundError: No module named 'isaacsim.core'" のエラーがでないように、
# IsaacSim 関連の import 文は AppLauncher の後に記載する必要がある
# ------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.ridgeback_franka import (  # isort:skip
    RIDGEBACK_FRANKA_PANDA_CFG
)

from isaaclab_assets import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort:skip


class FrankaSceneCfg(InteractiveSceneCfg):
    # 地面を配置
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # 照明を配置
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
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
            pos=(0.35, 0.0, 1.50),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
        # NOTE: Model was trained with rgb only
        data_types=["rgb"],
        # data_types=["rgb", "depth"],
        height=256,
        width=256,
    )

    # ロボットを配置
    # robot = ArticulationCfg(
    #     prim_path="/World/Robot",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka.usd",
    #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    #         activate_contact_sensors=False,
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         joint_pos={
    #             "panda_joint1": 0.0,
    #             "panda_joint2": -0.569,
    #             "panda_joint3": 0.0,
    #             "panda_joint4": -2.810,
    #             "panda_joint5": 0.0,
    #             "panda_joint6": 3.037,
    #             "panda_joint7": 0.741,
    #             "panda_finger_joint.*": 0.04,
    #         },
    #     ),
    #     actuators={
    #         "panda_shoulder": ImplicitActuatorCfg(
    #             joint_names_expr=["panda_joint[1-4]"],
    #             effort_limit_sim=87.0,
    #             velocity_limit_sim=2.175,
    #             stiffness=80.0,
    #             damping=4.0,
    #         ),
    #         "panda_forearm": ImplicitActuatorCfg(
    #             joint_names_expr=["panda_joint[5-7]"],
    #             effort_limit_sim=12.0,
    #             velocity_limit_sim=2.61,
    #             stiffness=80.0,
    #             damping=4.0,
    #         ),
    #         "panda_hand": ImplicitActuatorCfg(
    #             joint_names_expr=["panda_finger_joint.*"],
    #             effort_limit_sim=200.0,
    #             velocity_limit_sim=0.2,
    #             stiffness=2e3,
    #             damping=1e2,
    #         ),
    #     },
    #     soft_joint_pos_limit_factor=1.0,
    # )

    # robot = FRANKA_PANDA_CFG.replace(
    #     prim_path="/World/Robot",
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 1.0),
    #     ),
    # )

    robot_right = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
        ),
    )

    robot_left = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
        ),
    )

    # robot = RIDGEBACK_FRANKA_PANDA_CFG.replace(
    #     prim_path="/World/Robot",
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 1.0),
    #     ),
    # )


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
# シミュレーション実行
# ------------------------------------------------------------
sim_cfg = sim_utils.SimulationCfg(device=args.device)
sim = sim_utils.SimulationContext(sim_cfg)

# シーンを作成
scene_cfg = FrankaSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
scene = InteractiveScene(scene_cfg)
robot_right = scene["robot_right"]
robot_left = scene["robot_left"]
sensor_camera = scene["sensor_camera"]

print(f"シーン作成完了: {scene}")
print(f"robot_right: {vars(robot_right)}")
print(f"robot_left: {vars(robot_left)}")
print(f"sensor_camera: {vars(sensor_camera)}")

# カメラを配置
sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

# シミュレーションをリセット
sim.reset()

# ロボットのジョイント設定
print(f"ジョイント一覧: {robot_right.joint_names}")
MAP_JOINT_NAME_TO_IDX = {name: idx for idx, name in enumerate(robot_right.joint_names)}

# シミュレーション実行
sim_dt = sim.get_physics_dt()
sim_time = 0.0
count = 0
action_step = 0

print("シミュレーション開始...")

# action_buffer = {
#     "left_arm_eef_pos": [],
# }

while simulation_app.is_running():
    # ------------------------------------------------------------
    # 観測値（入力データ）の更新
    # ------------------------------------------------------------
    # ロボットの各関節のジョイント取得
    joint_pos_right = robot_right.data.joint_pos[0].cpu().numpy().astype(np.float32)
    joint_pos_left = robot_left.data.joint_pos[0].cpu().numpy().astype(np.float32)
    print(f"joint_pos_right.shape: {joint_pos_right.shape}")
    print(f"joint_pos_left.shape: {joint_pos_left.shape}")

    # ロボットの各関節のジョイント速度
    joint_vel_right = robot_right.data.joint_vel[0].cpu().numpy().astype(np.float32)
    joint_vel_left = robot_left.data.joint_vel[0].cpu().numpy().astype(np.float32)

    # End Effector （ロボットアームの先端部分に取り付けられた作業ツール）とグリッパーの位置と姿勢を取得
    right_arm_eef_pos = np.array(
        [
            joint_pos_right[MAP_JOINT_NAME_TO_IDX["panda_joint1"]],
            joint_pos_right[MAP_JOINT_NAME_TO_IDX["panda_joint2"]],
            joint_pos_right[MAP_JOINT_NAME_TO_IDX["panda_joint3"]],
        ],
        dtype=np.float64,
    ).reshape(1, 3)
    right_arm_eef_quat = np.array(
        [
            joint_pos_right[MAP_JOINT_NAME_TO_IDX["panda_joint4"]],
            joint_pos_right[MAP_JOINT_NAME_TO_IDX["panda_joint5"]],
            joint_pos_right[MAP_JOINT_NAME_TO_IDX["panda_joint6"]],
            joint_pos_right[MAP_JOINT_NAME_TO_IDX["panda_joint7"]],
        ],
        dtype=np.float64,
    ).reshape(1, 4)
    right_gripper_qpos = np.array(
        [
            joint_pos_right[MAP_JOINT_NAME_TO_IDX["panda_finger_joint1"]],
            joint_pos_right[MAP_JOINT_NAME_TO_IDX["panda_finger_joint2"]],
        ],
        dtype=np.float64,
    ).reshape(1, 2)

    left_arm_eef_pos = np.array(
        [
            joint_pos_left[MAP_JOINT_NAME_TO_IDX["panda_joint1"]],
            joint_pos_left[MAP_JOINT_NAME_TO_IDX["panda_joint2"]],
            joint_pos_left[MAP_JOINT_NAME_TO_IDX["panda_joint3"]],
        ],
        dtype=np.float64,
    ).reshape(1, 3)
    left_arm_eef_quat = np.array(
        [
            joint_pos_left[MAP_JOINT_NAME_TO_IDX["panda_joint4"]],
            joint_pos_left[MAP_JOINT_NAME_TO_IDX["panda_joint5"]],
            joint_pos_left[MAP_JOINT_NAME_TO_IDX["panda_joint6"]],
            joint_pos_left[MAP_JOINT_NAME_TO_IDX["panda_joint7"]],
        ],
        dtype=np.float64,
    ).reshape(1, 4)
    left_gripper_qpos = np.array(
        [
            joint_pos_left[MAP_JOINT_NAME_TO_IDX["panda_finger_joint1"]],
            joint_pos_left[MAP_JOINT_NAME_TO_IDX["panda_finger_joint2"]],
        ],
        dtype=np.float64,
    ).reshape(1, 2)

    # ロボットのカメラからの画像データ
    front_camera_image = sensor_camera.data.output["rgb"].cpu().numpy()
    cv2.imwrite(
        "front_camera.png", cv2.cvtColor(front_camera_image[0], cv2.COLOR_RGB2BGR)
    )

    # 推論用の入力データを作成
    observation = {
        "state.left_arm_eef_pos": left_arm_eef_pos,
        "state.left_arm_eef_quat": left_arm_eef_quat,
        "state.left_gripper_qpos": left_gripper_qpos,
        "state.right_arm_eef_pos": right_arm_eef_pos,
        "state.right_arm_eef_quat": right_arm_eef_quat,
        "state.right_gripper_qpos": right_gripper_qpos,
        "video.front_view": front_camera_image.astype(np.uint8),
        "video.left_wrist_view": front_camera_image.astype(np.uint8),
        "video.right_wrist_view": front_camera_image.astype(np.uint8),
        "task_description": [
            "move the red block to the other side, and move the hammer to the block's original position",
        ],
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

    # action を取得
    robot_right.set_joint_position_target(
        action_chunk["right_arm_eef_pos"],
        joint_ids=[
            MAP_JOINT_NAME_TO_IDX["panda_joint1"],
            MAP_JOINT_NAME_TO_IDX["panda_joint2"],
            MAP_JOINT_NAME_TO_IDX["panda_joint3"],
        ],
    )
    robot_right.set_joint_position_target(
        action_chunk["right_arm_eef_quat"],
        joint_ids=[
            MAP_JOINT_NAME_TO_IDX["panda_joint4"],
            MAP_JOINT_NAME_TO_IDX["panda_joint5"],
            MAP_JOINT_NAME_TO_IDX["panda_joint6"],
            MAP_JOINT_NAME_TO_IDX["panda_joint7"],
        ],
    )
    robot_right.set_joint_position_target(
        action_chunk["right_gripper_qpos"],
        joint_ids=[
            MAP_JOINT_NAME_TO_IDX["panda_finger_joint1"],
            MAP_JOINT_NAME_TO_IDX["panda_finger_joint2"],
        ],
    )

    robot_left.set_joint_position_target(
        action_chunk["left_arm_eef_pos"],
        joint_ids=[
            MAP_JOINT_NAME_TO_IDX["panda_joint1"],
            MAP_JOINT_NAME_TO_IDX["panda_joint2"],
            MAP_JOINT_NAME_TO_IDX["panda_joint3"],
        ],
    )
    robot_left.set_joint_position_target(
        action_chunk["left_arm_eef_quat"],
        joint_ids=[
            MAP_JOINT_NAME_TO_IDX["panda_joint4"],
            MAP_JOINT_NAME_TO_IDX["panda_joint5"],
            MAP_JOINT_NAME_TO_IDX["panda_joint6"],
            MAP_JOINT_NAME_TO_IDX["panda_joint7"],
        ],
    )
    robot_left.set_joint_position_target(
        action_chunk["left_gripper_qpos"],
        joint_ids=[
            MAP_JOINT_NAME_TO_IDX["panda_finger_joint1"],
            MAP_JOINT_NAME_TO_IDX["panda_finger_joint2"],
        ],
    )

    # シミュレーションステップ実行
    scene.write_data_to_sim()
    sim.step()
    sim_time += sim_dt
    count += 1
    scene.update(sim_dt)

# シミュレーション終了
simulation_app.close()
