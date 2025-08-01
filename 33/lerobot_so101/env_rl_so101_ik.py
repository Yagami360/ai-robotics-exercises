"""LeRobot SO-101 ロボットを使用したIK（逆運動学）対応環境の設定

このモジュールは Isaac Lab の ManagerBasedRLEnv を使用し、
teleoperation対応のSE(3)エンドエフェクター制御環境を定義します。
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from .env_rl_so101 import LeRobotSO101StackCubeRLEnvCfg

@configclass
class LeRobotSO101StackCubeIKRelEnvCfg(LeRobotSO101StackCubeRLEnvCfg):
    """LeRobot SO-101 ロボット用のIK（逆運動学）対応環境設定
    この設定では：
    - 関節角度制御ではなく、SE(3)エンドエフェクター制御を使用
    - Teleoperation（遠隔操作）対応
    - Differential IK（微分逆運動学）コントローラーを使用
    """

    def __post_init__(self):
        # 親クラスの初期化を実行
        super().__post_init__()

        # SO101ロボット用のIKアクション設定
        # 注意: SO101の関節名とエンドエフェクター名は実際のロボット仕様に合わせて調整が必要
        from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
        from isaaclab.managers import ActionTermCfg

        # エンドエフェクター制御用のアクション設定（実際のSO101関節名を使用）
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan",
                "shoulder_lift", 
                "elbow_flex",
                "wrist_flex",
                "wrist_roll"
            ],
            body_name="gripper",  # エンドエフェクター名（実際の名前に調整）
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls"  # Damped Least Squares法
            ),
            scale=0.5,
            # エンドエフェクターのオフセット（必要に応じて調整）
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0]
            ),
        )

        # グリッパー制御も追加（バイナリアクション）
        from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg

        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],  # 実際のSO101グリッパー関節名
            open_command_expr={"gripper": 0.04},   # 開く時の値
            close_command_expr={"gripper": 0.0},   # 閉じる時の値
        )
        print(f"[INFO]: IK環境設定が適用されました - SE(3)エンドエフェクター制御対応")
        print(f"[INFO]: 注意: 関節名とエンドエフェクター名は実際のSO101ロボット仕様に合わせて調整が必要です")
