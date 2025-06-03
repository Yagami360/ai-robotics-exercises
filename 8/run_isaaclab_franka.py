import argparse
import os

parser = argparse.ArgumentParser(description="Simple Simulation")
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
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ------------------------------------------------------------
# VNC サーバー用のディスプレイ設定
# ------------------------------------------------------------
if args.use_vnc:
    os.environ["DISPLAY"] = ":1"

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
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


class SimpleSceneCfg(InteractiveSceneCfg):
    # 地面を配置
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # 照明を配置
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # オブジェクト（ロボット）を配置
    # NOTE: 利用可能なロボット一覧は、以下のページを参照
    # https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_robots.html
    robot = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


# ------------------------------------------------------------
# シミュレーション実行
# ------------------------------------------------------------
sim_cfg = sim_utils.SimulationCfg(device=args.device)
sim = sim_utils.SimulationContext(sim_cfg)

# シーンを作成
# NOTE: 先に SimulationContext でシミュレーターを初期化してからシーンを作成する必要がある
scene_cfg = SimpleSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
scene = InteractiveScene(scene_cfg)
print(f"scene: {scene}")

# カメラを配置
sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

# シミュレーションをリセット
sim.reset()

# シミュレーションを実行
sim_dt = sim.get_physics_dt()
sim_time = 0.0
count = 0

while simulation_app.is_running():
    scene.write_data_to_sim()
    if count % 100 == 0:
        print(f"sim_time: {sim_time}")

        print(f"scene['robot']: {vars(scene['robot'])}")
        # {'cfg':
        #     ArticulationCfg(
        #         class_type=<class 'isaaclab.assets.articulation.articulation.Articulation'>,
        #         prim_path='/World/Robot',
        #         spawn=UsdFileCfg(
        #             func=<function spawn_from_usd at 0x7aca8f5c3640>,
        #             visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=None, kinematic_enabled=None, disable_gravity=False, linear_damping=None, angular_damping=None, max_linear_velocity=None, max_angular_velocity=None, max_depenetration_velocity=5.0, max_contact_impulse=None, enable_gyroscopic_forces=None, retain_accelerations=None, solver_position_iteration_count=None, solver_velocity_iteration_count=None, sleep_threshold=None, stabilization_threshold=None), collision_props=None,
        #             activate_contact_sensors=False, scale=None,
        #             articulation_props=ArticulationRootPropertiesCfg(
        #                 articulation_enabled=None, enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0, sleep_threshold=None,
        #                 stabilization_threshold=None, fix_root_link=None), fixed_tendons_props=None,
        #                 joint_drive_props=None, visual_material_path='material', visual_material=None,
        #                 usd_path='http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/FourierIntelligence/GR-1/GR1_T1.usd',
        #                 variants=None
        #             ),
        #             init_state=ArticulationCfg.InitialStateCfg(
        #                 pos=(0.0, 0.0, 0.0),
        #                 rot=(1.0, 0.0, 0.0, 0.0),
        #                 lin_vel=(0.0, 0.0, 0.0),
        #                 ang_vel=(0.0, 0.0, 0.0),
        #                 joint_pos={'.*': 0.0},
        #                 joint_vel={'.*': 0.0}
        #             ),
        #             collision_group=0, debug_vis=False, articulation_root_prim_path=None,
        #             soft_joint_pos_limit_factor=1.0,
        #             actuators={
        #                 'body': ImplicitActuatorCfg(
        #                     class_type=<class 'isaaclab.actuators.actuator_pd.ImplicitActuator'>,
        #                     joint_names_expr=['.*'],
        #                     effort_limit=300.0,
        #                     velocity_limit=10.0, effort_limit_sim=300.0, velocity_limit_sim=10.0, stiffness=40.0, damping=10.0, armature=None, friction=None
        #                 )
        #             }
        #         ),
        #         '_is_initialized': True,
        #         '_initialize_handle': <carb.events._events.ISubscription object at 0x7aca815f6230>,
        #         '_invalidate_initialize_handle': <carb.events._events.ISubscription object at 0x7aca819caf70>,
        #         '_debug_vis_handle': None,
        #         '_backend': 'torch',
        #         '_device': 'cuda:0',
        #         '_physics_sim_view': <omni.physics.tensors.impl.api.SimulationView object at 0x7aca7a5c01c0>,
        #         '_root_physx_view': <omni.physics.tensors.impl.api.ArticulationView object at 0x7aca7a5f4310>,
        #         '_data': <isaaclab.assets.articulation.articulation_data.ArticulationData object at 0x7aca7a9e3790>,
        #         '_ALL_INDICES': tensor([0], device='cuda:0'),
        #         'has_external_wrench': False,
        #         '_external_force_b': tensor([[[0., 0., 0.],
        #             ...
        #             [0., 0., 0.],
        #             [0., 0., 0.]]], device='cuda:0'),
        #         '_external_torque_b': tensor([[[0., 0., 0.],
        #             [0., 0., 0.],
        #             ...,
        #             [0., 0., 0.]]], device='cuda:0'),
        #         '_joint_pos_target_sim': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0'),
        #         '_joint_vel_target_sim': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0'), '_joint_effort_target_sim': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0'), 'actuators': {'body': <isaaclab.actuators.actuator_pd.ImplicitActuator object at 0x7aca7a9e0610>},
        #         '_has_implicit_actuators': True,
        #         '_fixed_tendon_names': []
        #     }

    sim.step()
    sim_time += sim_dt
    count += 1
    scene.update(sim_dt)

# シミュレーションを終了
simulation_app.close()
