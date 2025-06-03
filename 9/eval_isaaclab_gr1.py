import argparse
import os

import numpy as np
import torch

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="GR-1 Robot Simulation with Isaac-GR00T")
# parser.add_argument(
#     "--dataset_path",
#     type=str,
#     default="../Isaac-GR00T/demo_data/robot_sim.PickNPlace",
#     help="Dataset path for GR00T",
# )
parser.add_argument(
    "--model_path", type=str, default="nvidia/GR00T-N1-2B", help="GR00T model path"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument("--use_vnc", type=bool, default=True, help="Use VNC server")

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

# ------------------------------------------------------------
# VNC サーバー用のディスプレイ設定
# ------------------------------------------------------------
if args.use_vnc:
    os.environ["DISPLAY"] = ":1"

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
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


class GR1SceneCfg(InteractiveSceneCfg):
    # 地面を配置
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # 照明を配置
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # センサー：ロボット頭部にカメラを追加
    sensor_camera = CameraCfg(
        prim_path="/World/Robot/head_link/Camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
        data_types=["rgb", "depth"],
        height=256,
        width=256,
    )

    # GR-1ロボットを配置
    robot = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FourierIntelligence/GR-1/GR1_T1.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 重量を無効化
                # NOTE: 本学習済みモデルは、腕の行動ベクトルのみ出力するモデルであり、重量を有効化するとロボットが倒れるため
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            # T ポーズ
            joint_pos={
                # 左腕をTポーズ
                "l_shoulder_pitch": 0.0,
                "l_shoulder_roll": 1.57,
                "l_shoulder_yaw": 0.0,
                "l_elbow_pitch": 0.0,
                "l_wrist_yaw": 0.0,
                "l_wrist_roll": 0.0,
                "l_wrist_pitch": 0.0,
                # 右腕をTポーズ
                "r_shoulder_pitch": 0.0,
                "r_shoulder_roll": -1.57,
                "r_shoulder_yaw": 0.0,
                "r_elbow_pitch": 0.0,
                "r_wrist_yaw": 0.0,
                "r_wrist_roll": 0.0,
                "r_wrist_pitch": 0.0,
                # 脚部は直立姿勢
                "l_hip_yaw": 0.0,
                "l_hip_roll": 0.0,
                "l_hip_pitch": 0.0,
                "l_knee_pitch": 0.0,
                "l_ankle_pitch": 0.0,
                "l_ankle_roll": 0.0,
                "r_hip_yaw": 0.0,
                "r_hip_roll": 0.0,
                "r_hip_pitch": 0.0,
                "r_knee_pitch": 0.0,
                "r_ankle_pitch": 0.0,
                "r_ankle_roll": 0.0,
                # 腰
                "waist_yaw": 0.0,
                "waist_pitch": 0.0,
                "waist_roll": 0.0,
                # 頭部
                "head_yaw": 0.0,
                "head_roll": 0.0,
                "head_pitch": 0.0,
                # その他のジョイントはデフォルト値
                # ".*": 0.0,
            },
        ),
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    "l_shoulder.*",
                    "r_shoulder.*",
                    "l_elbow.*",
                    "r_elbow.*",
                    "l_wrist.*",
                    "r_wrist.*",
                ],
                effort_limit_sim=300.0,
                velocity_limit_sim=10.0,
                stiffness=80.0,
                damping=20.0,
            ),
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    "l_hip.*",
                    "r_hip.*",
                    "l_knee.*",
                    "r_knee.*",
                    "l_ankle.*",
                    "r_ankle.*",
                ],
                effort_limit_sim=500.0,
                velocity_limit_sim=5.0,
                stiffness=100.0,
                damping=30.0,
            ),
            "torso_head": ImplicitActuatorCfg(
                joint_names_expr=["waist.*", "head.*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=3.0,
                stiffness=60.0,
                damping=15.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


# ------------------------------------------------------------
# Isaac-GR00T モデル定義
# ------------------------------------------------------------
import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

data_config = DATA_CONFIG_MAP["gr1_arms_only"]

policy = Gr00tPolicy(
    model_path=args.model_path,
    modality_config=data_config.modality_config(),
    modality_transform=data_config.transform(),
    embodiment_tag=EmbodimentTag.GR1,
    device=args.device,
)

# サンプルデータセットからモデルへの入力データの形式を確認
# dataset = LeRobotSingleDataset(
#     dataset_path=args.dataset_path,
#     modality_configs=data_config.modality_config(),
#     transforms=None,
#     embodiment_tag=EmbodimentTag.GR1,
# )
# observation = dataset[0]
# print(f"observation keys: {observation.keys()}")
# for key, value in observation.items():
#     if hasattr(value, 'shape'):
#         print(f"{key}: shape={value.shape}, dtype={value.dtype}")
#     else:
#         print(f"{key}: {type(value)}, value={value}")

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
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow_pitch",
    "l_wrist_yaw",
    "l_wrist_roll",
    "l_wrist_pitch",
]

right_arm_joint_names = [
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow_pitch",
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch",
]
left_hand_joint_names = [
    "l_wrist_yaw",
    "l_wrist_roll",
    "l_wrist_pitch"
]
right_hand_joint_names = [
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch"
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

print(f"左腕ジョイント: {left_arm_joint_names}")
print(f"右腕ジョイント: {right_arm_joint_names}")
print(f"左腕ジョイントID: {left_arm_joint_ids}")
print(f"右腕ジョイントID: {right_arm_joint_ids}")
print(f"左手ジョイント: {left_hand_joint_names}")
print(f"右手ジョイント: {right_hand_joint_names}")
print(f"左手ジョイントID: {left_hand_joint_ids}")
print(f"右手ジョイントID: {right_hand_joint_ids}")

# シミュレーション実行
sim_dt = sim.get_physics_dt()
sim_time = 0.0
count = 0
action_step = 0

print("シミュレーション開始...")

while simulation_app.is_running():
    # 一定ステップごとにGR00Tで推論を実行
    # TODO: アクションチャンクを使用して、より柔軟な推論を実行
    if count % 100 == 0:
        print(f"シミュレーション時間: {sim_time:.2f}秒")

        # ------------------------------------------------------------
        # 入力データを設定
        # ------------------------------------------------------------
        # ロボットの各関節のジョイント位置
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

        # ロボットのカメラからの画像データ
        camera_data = sensor_camera.data
        rgb_image = camera_data.output["rgb"][0].cpu().numpy()
        camera_image = (rgb_image * 255).astype(np.uint8)
        camera_image = camera_image.reshape(1, 256, 256, 3)

        # 推論用の入力データを作成
        observation = {
            "state.left_arm": left_arm_state,
            "state.right_arm": right_arm_state,
            "state.left_hand": left_hand_state,
            "state.right_hand": right_hand_state,
            "video.ego_view": camera_image,
            "task_description": [
                "pick the pear from the counter and place it in the plate",
                # "shake the right hand",
            ],
        }
        print(f"observation: {observation}")

        # ------------------------------------------------------------
        # 推論処理
        # ------------------------------------------------------------
        with torch.inference_mode():
            # Isaac-GR00T モデルで推論
            action_chunk = policy.get_action(observation)
            print(f"action_chunk: {action_chunk}")
            # action_chunk: {
            #     'action.left_arm': array([[ 4.81812954e-02,  2.75385708e-01,  6.35790825e-02,
            #         -1.90729749e+00,  5.06091118e-03,  8.64368677e-02,
            #          1.70541644e-01],
            #         ...,
            #        [-2.77497768e-02,  3.04946452e-01, -5.54144382e-02,
            #         -1.79736257e+00,  1.15072966e-01,  2.42852449e-01,
            #          1.73459411e-01]]),
            #     'action.right_arm': array([[ 4.48875427e-02, -1.52109146e-01,  4.42804098e-02,
            #         -2.06703615e+00,  2.35692501e-01,  5.76206446e-02,
            #          5.29288054e-02],
            #         ...,
            #        [ 3.28695774e-02, -4.15798664e-01, -9.25958157e-03,
            #         -1.77494442e+00,  2.32261896e-01,  1.49469614e-01,
            #         -9.03737545e-03]]),
            #     'action.left_hand': array([[-0.03844249, -0.05444551, -0.03831387, -0.03725791, -0.06225586,
            #          0.05859375],
            #         ...,
            #        [-0.02465153, -0.04116249, -0.03471994, -0.0216043 , -0.02069092,
            #          0.02929688]]),
            #     'action.right_hand': array([[-1.48800468, -1.48742819, -1.48929691, -1.47856259, -2.953125  ,
            #          2.86523438],
            #         ...,
            #        [-1.5       , -1.5       , -1.5       , -1.4892813 , -2.98828125,
            #          2.95898438]])
            # }

        # ------------------------------------------------------------
        # 推論結果の行動（action）をロボットに適用
        # ------------------------------------------------------------
        # 左腕アクション
        left_arm_action = torch.tensor(
            action_chunk["action.left_arm"][0],
            device=args.device,
            dtype=torch.float32,
        )
        robot.set_joint_position_target(
            left_arm_action, joint_ids=left_arm_joint_ids
        )

        # 右腕アクション
        right_arm_action = torch.tensor(
            action_chunk["action.right_arm"][0],
            device=args.device,
            dtype=torch.float32,
        )
        robot.set_joint_position_target(
            right_arm_action, joint_ids=right_arm_joint_ids
        )

        # 左手アクション
        if len(left_hand_joint_ids) > 0:
            left_hand_action_full = action_chunk["action.left_hand"][0]
            left_hand_action = torch.tensor(
                # 6次元から3次元に調整
                left_hand_action_full[:len(left_hand_joint_ids)],
                device=args.device,
                dtype=torch.float32,
            )
            robot.set_joint_position_target(
                left_hand_action, joint_ids=left_hand_joint_ids
            )

        # 右手アクション
        if len(right_hand_joint_ids) > 0:
            right_hand_action_full = action_chunk["action.right_hand"][0]
            right_hand_action = torch.tensor(
                # 6次元から3次元に調整
                right_hand_action_full[:len(right_hand_joint_ids)],
                device=args.device,
                dtype=torch.float32,
            )
            robot.set_joint_position_target(
                right_hand_action, joint_ids=right_hand_joint_ids
            )

        action_step += 1

    # シミュレーションステップ実行
    scene.write_data_to_sim()
    sim.step()
    sim_time += sim_dt
    count += 1
    scene.update(sim_dt)

# シミュレーション終了
simulation_app.close()
