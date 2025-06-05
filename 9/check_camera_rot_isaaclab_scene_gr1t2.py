import argparse
import os

import cv2
import numpy as np
import torch

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="GR-1 Robot Simulation")
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
                kinematic_enabled=True,  # 固定オブジェクト
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.6, 0.4),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 1.21),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, -0.8, 0.2),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.35, -0.07, 1.28),
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
            pos=(0.50, -0.05, 1.28),
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
            pos=(0.35, 0.0, 1.35),
            # オイラー角 X,Y,Z = (180.0, 0.0, -73.0) 相当のクオータニオン (w,x,y,z) になる
            rot=(0.0, 1.0, 0.0, 0.0),
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

# シミュレーション実行
sim_dt = sim.get_physics_dt()
sim_time = 0.0
count = 0

print("シミュレーション開始...")

camera_rot_x = 0.0
camera_rot_y = 0.0
camera_rot_z = 0.0
camera_rot_w = 1.0

while simulation_app.is_running():
    for camera_rot_x in np.arange(0.0, 1.0, 0.1):
        for camera_rot_y in np.arange(0.0, 1.0, 0.1):
            for camera_rot_z in np.arange(0.0, 1.0, 0.1):
                for camera_rot_w in np.arange(0.0, 1.0, 0.1):
                    sensor_camera.cfg.offset.rot = (
                        camera_rot_w,
                        camera_rot_x,
                        camera_rot_y,
                        camera_rot_z,
                    )
                    camera_pos = sensor_camera.cfg.offset.pos
                    camera_rot = sensor_camera.cfg.offset.rot
                    print(f"camera_pos: {camera_pos}, camera_rot: {camera_rot}")

                    # ロボットのカメラからの画像データ
                    camera_image = sensor_camera.data.output["rgb"].cpu().numpy()
                    cv2.imwrite(
                        f"robot-cameras/robot-camera-pos-x{camera_pos[0]}-y{camera_pos[1]}-z{camera_pos[2]}-rot-w{camera_rot[0]}-x{camera_rot[1]}-y{camera_rot[2]}-z{camera_rot[3]}.png",
                        cv2.cvtColor(camera_image[0], cv2.COLOR_RGB2BGR),
                    )

                    # シミュレーションステップ実行
                    scene.write_data_to_sim()
                    sim.step()
                    sim_time += sim_dt
                    count += 1
                    scene.update(sim_dt)

# シミュレーション終了
simulation_app.close()
