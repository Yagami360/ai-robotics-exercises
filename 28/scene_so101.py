import argparse
import os

# ------------------------------------------------------------
# シミュレーターアプリ作成
# ------------------------------------------------------------
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="SO-ARM101ロボットをIsaac Simで配置するスクリプト"
)
parser.add_argument(
    "--usd_path",
    type=str,
    default="../assets/SO101/so101_new_calib/so101_new_calib.usd",
    help="出力USDファイルのパス"
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="配置する環境の数"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


class SO101SceneCfg(InteractiveSceneCfg):
    def __init__(self, usd_path, num_envs=1, env_spacing=2.0):
        super().__init__(num_envs, env_spacing)

        # 地面を配置
        self.ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg()
        )

        # 照明を配置
        self.dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )

        # SO-ARM101ロボットを配置
        self.so101_robot = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    # SO-ARM101の実際のジョイント名
                    "shoulder_pan": 0.0,
                    "shoulder_lift": 0.0,
                    "elbow_flex": 0.0,
                    "wrist_flex": 0.0,
                    "wrist_roll": 0.0,
                    "gripper": 0.0,
                },
                pos=(0.0, 0.0, 0.0),  # ロボットの初期位置
            ),
            # TODO: AttributeError: 'Articulation' object has no attribute 'has_external_wrench' のエラーを要修正
            actuators={
                "arm_actuator": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],  # 全ジョイントを制御
                    effort_limit_sim=100.0,
                    velocity_limit_sim=100.0,
                    stiffness=10000.0,
                    damping=100.0,
                ),
            },
        ).replace(
            prim_path="{ENV_REGEX_NS}/SO101_Robot"
        )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """シミュレーションを実行"""
    print("[INFO]: シミュレーション開始...")

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # シーンのデータをシミュレーターに書き込み
        scene.write_data_to_sim()

        # シミュレーションをステップ実行
        sim.step(render=True)

        # シミュレーション時間を更新
        sim_time += sim_dt
        count += 1

        # 定期的に情報を出力
        if count % 100 == 0:
            print(f"[INFO]: シミュレーション時間: {sim_time:.2f}s, ステップ数: {count}")

        # シーンを更新
        scene.update(sim_dt)


def main():
    """メイン関数"""
    # 引数の出力
    for arg in vars(args_cli):
        print(f"{arg}: {getattr(args_cli, arg)}")

    # USDファイルが存在するかチェック
    if not os.path.exists(args_cli.usd_path):
        print(f"エラー: USDファイルが見つかりません: {args_cli.usd_path}")
        return
    else:
        print(f"USDファイルを確認しました: {args_cli.usd_path}")

    # ------------------------------------------------------------
    # シミュレーション実行
    # ------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # シーンを作成
    # NOTE: 先に SimulationContext でシミュレーターを初期化してからシーンを作成する必要がある
    scene_cfg = SO101SceneCfg(args_cli.usd_path, num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    print(f"scene: {scene}")

    # カメラを配置
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # シミュレーションをリセット
    sim.reset()

    print("[INFO]: セットアップ完了...")

    # シミュレーションを実行
    run_simulator(sim, scene)

    # シミュレーションを終了
    simulation_app.close()


if __name__ == "__main__":
    main()
