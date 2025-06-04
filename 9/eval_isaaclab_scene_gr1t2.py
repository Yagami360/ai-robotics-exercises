import argparse
import os

import cv2
import numpy as np
import torch

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="GR-1 Robot Simulation with Isaac-GR00T")
parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1-2B")
# parser.add_argument("--model_path", type=str, default="../checkpoints/gr00t/checkpoint-1000/")
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
            size=(0.6, 0.4, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # 固定オブジェクト
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.6, 0.4),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.4, 0.0, 1.0),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, -0.8, 0.2),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.25, -0.07, 1.05),
        ),
    )

    # プレート（皿）を配置
    plate = AssetBaseCfg(
        prim_path="/World/Plate",
        spawn=sim_utils.CylinderCfg(
            radius=0.06,
            height=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # 固定オブジェクト
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.25, 0.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.40, -0.05, 1.05),
        ),
    )

    # センサー：ロボット頭部にカメラを追加
    sensor_camera = CameraCfg(
        prim_path="/World/Robot/head_link/Camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=300.0,
            horizontal_aperture=40.0,
            vertical_aperture=40.0,
            clipping_range=(0.05, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 1.5),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
        data_types=["rgb", "depth"],
        height=256,
        width=256,
    )

    # GR-1-T2 ロボットを配置
    robot: ArticulationCfg = GR1T2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
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
config_key = "gr1_arms_only"
# config_key = "gr1_arms_waist"
data_config = DATA_CONFIG_MAP[config_key]

policy = Gr00tPolicy(
    model_path=args.model_path,
    modality_config=data_config.modality_config(),
    modality_transform=data_config.transform(),
    embodiment_tag=EmbodimentTag.GR1 if args.model_path == "nvidia/GR00T-N1-2B" else EmbodimentTag.NEW_EMBODIMENT,
    device=args.device,
)

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
print(f"robot: {vars(robot)}")
print(f"sensor_camera: {vars(sensor_camera)}")

# カメラを配置
sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

# シミュレーションをリセット
sim.reset()

# GR-1ロボットのジョイント設定
print(f"ジョイント一覧: {robot.joint_names}")

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
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_distal_joint",
]

right_hand_joint_names = [
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_distal_joint",
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
        # 必要な観測値を更新
        joint_pos = robot.data.joint_pos[0].cpu().numpy().astype(np.float32)

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
        # left_hand_state = np.zeros((1, 11), dtype=np.float32)
        # right_hand_state = np.zeros((1, 11), dtype=np.float32)

        # 腰部のジョイント位置
        if config_key == "gr1_arms_waist":
            waist_state = np.zeros((1, 3), dtype=np.float32)
            if len(waist_joint_ids) >= 3:
                waist_state[0] = joint_pos[waist_joint_ids[:3]]
            elif len(waist_joint_ids) > 0:
                waist_state[0, : len(waist_joint_ids)] = joint_pos[waist_joint_ids]

        # ロボットのカメラからの画像データ
        camera_data = sensor_camera.data
        rgb_image = camera_data.output["rgb"][0].cpu().numpy()
        camera_image = (rgb_image * 255).astype(np.uint8)
        camera_image = camera_image.reshape(1, 256, 256, 3)

        image_to_save = camera_image[0]
        cv2.imwrite("robot_camera.png", cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))

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
        # print(f"observation: {observation}")

        with torch.inference_mode():
            action_chunk = policy.get_action(observation)
        # 各アクションをバッファに格納（list化してpop(0)で取り出せるように）
        for k in action_buffer.keys():
            action_buffer[k] = list(action_chunk[f"action.{k}"])

    # バッファから1ステップ分のアクションを取り出す
    left_arm_action = torch.tensor(
        action_buffer["left_arm"].pop(0), device=args.device, dtype=torch.float32
    )
    right_arm_action = torch.tensor(
        action_buffer["right_arm"].pop(0), device=args.device, dtype=torch.float32
    )
    left_hand_action = torch.tensor(
        action_buffer["left_hand"].pop(0), device=args.device, dtype=torch.float32
    )
    right_hand_action = torch.tensor(
        action_buffer["right_hand"].pop(0), device=args.device, dtype=torch.float32
    )
    if config_key == "gr1_arms_waist":
        waist_action = torch.tensor(
            action_buffer["waist"].pop(0), device=args.device, dtype=torch.float32
        )

    # ロボットにアクションを適用
    robot.set_joint_position_target(
        left_arm_action,
        joint_ids=left_arm_joint_ids
    )
    robot.set_joint_position_target(
        right_arm_action,
        joint_ids=right_arm_joint_ids
    )
    robot.set_joint_position_target(
        left_hand_action,
        # joint_ids=left_hand_joint_ids,
        joint_ids=left_hand_joint_ids[:len(left_hand_action)]
    )
    robot.set_joint_position_target(
        right_hand_action,
        # joint_ids=right_hand_joint_ids,
        joint_ids=right_hand_joint_ids[:len(right_hand_action)]
    )
    if config_key == "gr1_arms_waist":
        robot.set_joint_position_target(
            waist_action,
            joint_ids=waist_joint_ids[:len(waist_action)]
        )

    # シミュレーションステップ実行
    scene.write_data_to_sim()
    sim.step()
    sim_time += sim_dt
    count += 1
    scene.update(sim_dt)

# シミュレーション終了
simulation_app.close()
