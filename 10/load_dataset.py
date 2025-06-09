import argparse
import os

import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # "--dataset_path", type=str, default="../datasets/single_panda_gripper.OpenSingleDoor"
        # "--dataset_path", type=str, default="../datasets/bimanual_panda_gripper.Transport"
        "--dataset_path",
        type=str,
        default="../datasets/bimanual_panda_hand.LiftTray",
    )
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # data_config = DATA_CONFIG_MAP["bimanual_panda_gripper"]
    data_config = DATA_CONFIG_MAP["bimanual_panda_hand"]

    # LeRobot のデータセットを読み込む
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=data_config.modality_config(),
        transforms=None,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )

    # データセットの概要を print する
    print("dataset:", dataset)
    print("dataset[0].keys():", dataset[0].keys())
    for k in dataset[0].keys():
        if isinstance(dataset[0][k], list):
            print(f"dataset[0][{k}]: {dataset[0][k]}")
        else:
            print(
                f"dataset[0][{k}] shape: {dataset[0][k].shape}, dtype: {dataset[0][k].dtype}, min: {dataset[0][k].min()}, max: {dataset[0][k].max()}"
            )

    # dataset: bimanual_panda_gripper.Transport (409386 steps)
    # dataset[0].keys(): dict_keys(['video.right_wrist_view', 'video.left_wrist_view', 'video.front_view', 'state.right_arm_eef_pos', 'state.right_arm_eef_quat', 'state.right_gripper_qpos', 'state.left_arm_eef_pos', 'state.left_arm_eef_quat', 'state.left_gripper_qpos', 'action.right_arm_eef_pos', 'action.right_arm_eef_rot', 'action.right_gripper_close', 'action.left_arm_eef_pos', 'action.left_arm_eef_rot', 'action.left_gripper_close', 'annotation.human.action.task_description'])
    # dataset[0][video.right_wrist_view] shape: (1, 256, 256, 3), dtype: uint8, min: 0, max: 255
    # dataset[0][video.left_wrist_view] shape: (1, 256, 256, 3), dtype: uint8, min: 0, max: 255
    # dataset[0][video.front_view] shape: (1, 256, 256, 3), dtype: uint8, min: 0, max: 255
    # dataset[0][state.right_arm_eef_pos] shape: (1, 3), dtype: float64, min: -0.57683567345639, max: 1.0142335992269078
    # dataset[0][state.right_arm_eef_quat] shape: (1, 4), dtype: float64, min: -0.002330420961333981, max: 0.9971349109701646
    # dataset[0][state.right_gripper_qpos] shape: (1, 2), dtype: float64, min: -0.020833, max: 0.020833
    # dataset[0][state.left_arm_eef_pos] shape: (1, 3), dtype: float64, min: -0.1032882134309532, max: 1.020581108636769
    # dataset[0][state.left_arm_eef_quat] shape: (1, 4), dtype: float64, min: -0.007954808384067202, max: 0.998419789421527
    # dataset[0][state.left_gripper_qpos] shape: (1, 2), dtype: float64, min: -0.020833, max: 0.020833
    # dataset[0][action.right_arm_eef_pos] shape: (16, 3), dtype: float64, min: -0.24831672291498658, max: 0.5949151741849552
    # dataset[0][action.right_arm_eef_rot] shape: (16, 3), dtype: float64, min: -0.14456175706898633, max: 0.2845573415685707
    # dataset[0][action.right_gripper_close] shape: (16, 1), dtype: float64, min: 0.0, max: 0.0
    # dataset[0][action.left_arm_eef_pos] shape: (16, 3), dtype: float64, min: -1.0, max: 0.903452953119267
    # dataset[0][action.left_arm_eef_rot] shape: (16, 3), dtype: float64, min: -1.0, max: 0.17658973554480173
    # dataset[0][action.left_gripper_close] shape: (16, 1), dtype: float64, min: 0.0, max: 0.0
    # dataset[0][annotation.human.action.task_description]: ["move the red block to the other side, and move the hammer to the block's original position"]

    # 特定のエピソードと時間ステップのデータを print する
    # print("dataset[0]:", dataset[0])
