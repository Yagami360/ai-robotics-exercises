import argparse
import os
import sys
import time


# 環境変数を最初に設定
def setup_environment():
    """環境変数の事前設定"""
    # VNC環境でのディスプレイ設定
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":1"

    # OpenGL設定（VNC環境での3D描画対応）
    os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
    os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"

    # CUDA設定
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Isaac Sim関連の環境変数
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

    print(f"ディスプレイ設定: {os.environ.get('DISPLAY', 'Not set')}")


# 環境設定を最初に実行
setup_environment()

# Isaac Labs関連のインポート（環境設定後）
from isaaclab.app import AppLauncher


# AppLauncherを最初に初期化
def create_app_launcher(headless=False):
    """AppLauncherの作成"""
    # メインのargparserを作成
    parser = argparse.ArgumentParser(description="Isaac-GR00T Simulation")

    # 独自の引数を先に追加
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../Isaac-GR00T/demo_data/robot_sim.PickNPlace",
        help="データセットのパス",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="nvidia/GR00T-N1-2B",
        help="事前学習済みモデルのパス",
    )
    parser.add_argument("--seed", type=int, default=42, help="ランダムシード")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--num_steps", type=int, default=1000, help="シミュレーションステップ数"
    )
    parser.add_argument("--debug_mode", action="store_true", help="詳細ログを出力")

    # AppLauncherの引数を追加
    AppLauncher.add_app_launcher_args(parser)

    # 引数をパース
    args = parser.parse_args()

    # AppLauncherを作成
    app_launcher = AppLauncher(args)

    return app_launcher, args


# AppLauncherを事前に作成
app_launcher, main_args = create_app_launcher()
simulation_app = app_launcher.app

# Isaac Labs初期化後にGR00T関連をインポート
try:
    import gr00t
    import numpy as np
    import torch
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    print("GR00Tライブラリのインポートが完了しました")
except Exception as e:
    print(f"GR00Tライブラリのインポートでエラー: {e}")
    simulation_app.close()
    sys.exit(1)


class IsaacGR00TSimulation:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if args.gpu_id >= 0 else "cpu"
        self.policy = None
        self.dataset = None
        self.world = None
        self.robot = None

        # シード設定
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # データ設定とポリシーの初期化
        self.setup_dataset()
        self.setup_policy()

        # シミュレーションのセットアップ
        self.setup_simulation()

    def setup_dataset(self):
        """データセットのセットアップ"""
        try:
            print("データセットを初期化しています...")

            # データセットパスの確認
            if os.path.exists(self.args.dataset_path):
                print(f"データセットパス: {self.args.dataset_path}")

                # LeRobotSingleDatasetを使用してデータセットを読み込み
                self.dataset = LeRobotSingleDataset(
                    dataset_path=self.args.dataset_path, split="train"
                )

                print(f"データセットサイズ: {len(self.dataset)}")

                # サンプルデータの確認
                if len(self.dataset) > 0:
                    sample = self.dataset[0]
                    print("サンプルデータのキー:")
                    for key in sample.keys():
                        if isinstance(sample[key], np.ndarray):
                            print(f"  {key}: shape={sample[key].shape}")
                        else:
                            print(f"  {key}: {type(sample[key])}")

            else:
                print(f"データセットパスが見つかりません: {self.args.dataset_path}")
                # ダミーデータセットを作成
                self.create_dummy_dataset()

        except Exception as e:
            print(f"データセット初期化でエラー: {e}")
            self.create_dummy_dataset()

    def create_dummy_dataset(self):
        """ダミーデータセットの作成"""
        print("ダミーデータセットを作成します...")

        class DummyDataset:
            def __init__(self):
                self.data = []
                # ダミーデータを生成
                for i in range(10):
                    sample = {
                        "observation.images.top": np.random.randint(
                            0, 255, (224, 224, 3), dtype=np.uint8
                        ),
                        "observation.state": np.random.randn(47).astype(np.float32),
                        "action.left_arm": np.random.randn(7).astype(np.float32),
                        "action.right_arm": np.random.randn(7).astype(np.float32),
                        "action.left_hand": np.random.randn(6).astype(np.float32),
                        "action.right_hand": np.random.randn(6).astype(np.float32),
                    }
                    self.data.append(sample)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        self.dataset = DummyDataset()
        print(f"ダミーデータセットを作成しました（サイズ: {len(self.dataset)}）")

    def setup_policy(self):
        """ポリシーのセットアップ"""
        try:
            print("ポリシーを初期化しています...")

            # 利用可能なデータ設定を確認
            print("利用可能なデータ設定:")
            for key in DATA_CONFIG_MAP.keys():
                print(f"  - {key}")

            # 適切なデータ設定を選択
            data_config_key = "gr1_arms_only"  # 修正: 正しいキーを使用
            if data_config_key not in DATA_CONFIG_MAP:
                # フォールバック: 最初の利用可能な設定を使用
                data_config_key = list(DATA_CONFIG_MAP.keys())[0]
                print(f"フォールバック設定を使用: {data_config_key}")

            data_config = DATA_CONFIG_MAP[data_config_key]

            # ポリシーを初期化
            self.policy = Gr00tPolicy(
                model_name=self.args.model_path,
                data_config=data_config,
                device=self.device,
            )

            print(f"ポリシーの初期化が完了しました（モデル: {self.args.model_path}）")

        except Exception as e:
            print(f"ポリシー初期化でエラー: {e}")
            # ダミーポリシーを作成
            self.create_dummy_policy()

    def create_dummy_policy(self):
        """ダミーポリシーの作成"""
        print("ダミーポリシーを作成します...")

        class DummyPolicy:
            def get_action(self, observation):
                # ダミーアクションを生成
                return {
                    "action.left_arm": torch.randn(7),
                    "action.right_arm": torch.randn(7),
                    "action.left_hand": torch.randn(6),
                    "action.right_hand": torch.randn(6),
                }

        self.policy = DummyPolicy()
        print("ダミーポリシーを作成しました")

    def setup_simulation(self):
        """シミュレーション環境のセットアップ"""
        print("シミュレーション環境を初期化しています...")

        try:
            # Isaac Labsの新しいAPIを使用してシーンを作成
            import omni.usd
            from pxr import Gf, UsdGeom, UsdPhysics

            # 新しいステージを作成
            stage = omni.usd.get_context().new_stage()

            # 基本的なシーンセットアップ
            self.setup_basic_scene(stage)

            # ロボットを追加
            self.add_robot_to_scene(stage)

            # 物体を追加
            self.add_objects_to_scene(stage)

            print("シミュレーション環境の初期化が完了しました")

        except Exception as e:
            print(f"シミュレーションセットアップでエラー: {e}")
            print("シンプルなシミュレーション環境で続行します")
            self.setup_simple_scene()

    def setup_basic_scene(self, stage):
        """基本的なシーンのセットアップ"""
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        # ルートプリムを作成
        root_prim = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(root_prim)

        # 物理シーンを設定
        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

        # 照明を追加
        light = UsdGeom.DistantLight.Define(stage, "/World/DistantLight")
        light.CreateIntensityAttr(500)
        light.CreateAngleAttr(1)
        light.GetPrim().GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(315, 0, 0))

        # 地面を追加
        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.CreateSizeAttr(1.0)
        ground.AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.5))
        ground.AddScaleOp().Set(Gf.Vec3f(10, 10, 1))

        # 地面の物理設定
        ground_collision = UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

        print("基本シーンを作成しました")

    def add_robot_to_scene(self, stage):
        """ロボットをシーンに追加"""
        try:
            from pxr import Gf, UsdGeom

            # GR1ロボットの簡易表現を作成
            robot_group = UsdGeom.Xform.Define(stage, "/World/GR1_Robot")
            robot_group.AddTranslateOp().Set(Gf.Vec3f(0, 0, 1))

            # ベース（胴体）
            base = UsdGeom.Cylinder.Define(stage, "/World/GR1_Robot/Base")
            base.CreateHeightAttr(1.0)
            base.CreateRadiusAttr(0.3)
            base.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.5))

            # 左腕
            left_arm = UsdGeom.Cylinder.Define(stage, "/World/GR1_Robot/LeftArm")
            left_arm.CreateHeightAttr(0.6)
            left_arm.CreateRadiusAttr(0.05)
            left_arm.AddTranslateOp().Set(Gf.Vec3f(-0.4, 0, 0.8))
            left_arm.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 90))

            # 右腕
            right_arm = UsdGeom.Cylinder.Define(stage, "/World/GR1_Robot/RightArm")
            right_arm.CreateHeightAttr(0.6)
            right_arm.CreateRadiusAttr(0.05)
            right_arm.AddTranslateOp().Set(Gf.Vec3f(0.4, 0, 0.8))
            right_arm.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 90))

            # 左手
            left_hand = UsdGeom.Sphere.Define(stage, "/World/GR1_Robot/LeftHand")
            left_hand.CreateRadiusAttr(0.08)
            left_hand.AddTranslateOp().Set(Gf.Vec3f(-0.7, 0, 0.8))

            # 右手
            right_hand = UsdGeom.Sphere.Define(stage, "/World/GR1_Robot/RightHand")
            right_hand.CreateRadiusAttr(0.08)
            right_hand.AddTranslateOp().Set(Gf.Vec3f(0.7, 0, 0.8))

            # 頭部
            head = UsdGeom.Sphere.Define(stage, "/World/GR1_Robot/Head")
            head.CreateRadiusAttr(0.15)
            head.AddTranslateOp().Set(Gf.Vec3f(0, 0, 1.2))

            print("ロボットをシーンに追加しました")

        except Exception as e:
            print(f"ロボット追加でエラー: {e}")

    def add_objects_to_scene(self, stage):
        """操作対象の物体をシーンに追加"""
        try:
            from pxr import Gf, UsdGeom, UsdPhysics

            # テーブル
            table = UsdGeom.Cube.Define(stage, "/World/Table")
            table.CreateSizeAttr(1.0)
            table.AddTranslateOp().Set(Gf.Vec3f(1.0, 0, 0.4))
            table.AddScaleOp().Set(Gf.Vec3f(1.2, 0.8, 0.8))

            # 操作対象のボックス
            box = UsdGeom.Cube.Define(stage, "/World/TargetBox")
            box.CreateSizeAttr(0.1)
            box.AddTranslateOp().Set(Gf.Vec3f(1.0, 0, 0.85))

            # ボックスの物理設定
            box_rigid_body = UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
            box_collision = UsdPhysics.CollisionAPI.Apply(box.GetPrim())

            print("操作対象の物体をシーンに追加しました")

        except Exception as e:
            print(f"物体追加でエラー: {e}")

    def setup_simple_scene(self):
        """シンプルなシーンのセットアップ（フォールバック）"""
        try:
            import omni.usd
            from pxr import Gf, UsdGeom

            # 現在のステージを取得
            stage = omni.usd.get_context().get_stage()

            if stage:
                # シンプルなキューブを追加
                cube = UsdGeom.Cube.Define(stage, "/World/SimpleCube")
                cube.CreateSizeAttr(1.0)
                cube.AddTranslateOp().Set(Gf.Vec3f(0, 0, 1))

                print("シンプルなシーンを作成しました")

        except Exception as e:
            print(f"シンプルなシーン作成でエラー: {e}")

    def simulate_robot_actions(self):
        """ロボットアクションのシミュレーション"""
        try:
            # データセットからランダムなサンプルを取得
            sample_idx = np.random.randint(0, len(self.dataset))
            sample = self.dataset[sample_idx]

            # ポリシーからアクションを予測
            with torch.no_grad():
                action_chunk = self.policy.get_action(sample)

            # アクション情報を表示
            if self.args.debug_mode:
                print(f"アクションチャンクを生成しました:")
                for key, value in action_chunk.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, mean={value.mean():.4f}")
                    elif isinstance(value, np.ndarray):
                        print(f"  {key}: shape={value.shape}, mean={value.mean():.4f}")
                    else:
                        print(f"  {key}: {value}")

            return action_chunk

        except Exception as e:
            print(f"アクションシミュレーションでエラー: {e}")
            return None

    def update_robot_pose(self, action_chunk):
        """アクションに基づいてロボットの姿勢を更新"""
        try:
            import omni.usd
            from pxr import Gf

            stage = omni.usd.get_context().get_stage()
            if not stage:
                return

            # 左手の位置を更新
            if "action.left_arm" in action_chunk:
                left_hand_prim = stage.GetPrimAtPath("/World/GR1_Robot/LeftHand")
                if left_hand_prim.IsValid():
                    left_arm_action = action_chunk["action.left_arm"]
                    if isinstance(left_arm_action, torch.Tensor):
                        left_arm_action = left_arm_action.cpu().numpy()

                    # アクションを位置に変換（簡易的な変換）
                    x_offset = float(left_arm_action[0]) * 0.1
                    y_offset = float(left_arm_action[1]) * 0.1
                    z_offset = float(left_arm_action[2]) * 0.1

                    new_pos = Gf.Vec3f(-0.7 + x_offset, y_offset, 0.8 + z_offset)
                    left_hand_prim.GetAttribute("xformOp:translate").Set(new_pos)

            # 右手の位置を更新
            if "action.right_arm" in action_chunk:
                right_hand_prim = stage.GetPrimAtPath("/World/GR1_Robot/RightHand")
                if right_hand_prim.IsValid():
                    right_arm_action = action_chunk["action.right_arm"]
                    if isinstance(right_arm_action, torch.Tensor):
                        right_arm_action = right_arm_action.cpu().numpy()

                    # アクションを位置に変換（簡易的な変換）
                    x_offset = float(right_arm_action[0]) * 0.1
                    y_offset = float(right_arm_action[1]) * 0.1
                    z_offset = float(right_arm_action[2]) * 0.1

                    new_pos = Gf.Vec3f(0.7 + x_offset, y_offset, 0.8 + z_offset)
                    right_hand_prim.GetAttribute("xformOp:translate").Set(new_pos)

        except Exception as e:
            if self.args.debug_mode:
                print(f"ロボット姿勢更新でエラー: {e}")

    def run_simulation(self, num_steps=1000):
        """シミュレーションの実行"""
        print("シミュレーションを開始します...")

        try:
            step_count = 0
            action_update_interval = 50  # 50ステップごとにアクションを更新

            while simulation_app.is_running() and step_count < num_steps:
                # 定期的にポリシーからアクションを取得
                if step_count % action_update_interval == 0:
                    action_chunk = self.simulate_robot_actions()
                    if action_chunk:
                        print(f"ステップ {step_count}: 新しいアクションを取得しました")
                        # ロボットの姿勢を更新
                        self.update_robot_pose(action_chunk)

                # 進捗表示
                if step_count % 100 == 0:
                    print(
                        f"ステップ {step_count}/{num_steps}: シミュレーション実行中..."
                    )

                step_count += 1

                # フレームレート制御（VNC環境での負荷軽減）
                time.sleep(0.02)  # 50 FPS

            print("シミュレーションが完了しました")

        except Exception as e:
            print(f"シミュレーション実行でエラー: {e}")
            raise


def main():
    # 引数の表示
    print("=== 実行パラメータ ===")
    for arg in vars(main_args):
        print(f"{arg}: {getattr(main_args, arg)}")
    print("==================")

    # GPU設定
    if main_args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(main_args.gpu_id)
        print(f"GPU {main_args.gpu_id} を使用します")
    else:
        print("CPUモードで実行します")

    try:
        # シミュレーションの実行
        sim = IsaacGR00TSimulation(main_args)
        sim.run_simulation(num_steps=main_args.num_steps)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # アプリケーションの終了
        try:
            simulation_app.close()
            print("アプリケーションを正常に終了しました")
        except Exception as e:
            print(f"アプリケーション終了時にエラー: {e}")


if __name__ == "__main__":
    main()
