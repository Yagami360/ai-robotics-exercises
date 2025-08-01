"""SO101 ロボット環境の登録"""

import gymnasium as gym

##
# ｛LeRobot の SO-101 ロボット x Cube 積み重ね｝の Gymnasium 環境を登録
##

##
# Joint Position Control
##

gym.register(
    id="LeRobot-SO101-StackCube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "lerobot_so101.env_rl_so101:LeRobotSO101StackCubeRLEnvCfg",
    },
    disable_env_checker=True,
)

# 利用可能な環境のリスト
__all__ = [
    "LeRobot-SO101-StackCube-v0",
]
