# create_video_with_sim.py
import os
import sys
import cv2
import numpy as np
import asyncio
import nest_asyncio
nest_asyncio.apply()

from argparse import ArgumentParser, Namespace
from isaaclab.app import AppLauncher

OUTPUT_DIR = "datasets/generated_dataset"

def start_isaac_sim():
    print("Isaac Simを起動中...")

    # パーサーの設定
    parser = ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args([])

    # 設定
    config = {
        "task": "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
        "num_envs": 1,
        "generation_num_trials": 1,
        "input_file": "datasets/annotated_dataset.hdf5",
        "output_file": "datasets/generated_dataset.hdf5",
        "pause_subtask": False,
        "enable": "omni.kit.renderer.capture",
        "headless": True,
        "kit_args": "--headless --enable omni.videoencoding --no-window",
    }

    # 設定を適用
    args_dict = vars(args_cli)
    args_dict.update(config)
    args_cli = Namespace(**args_dict)

    # Isaac Simを起動
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    print("Isaac Sim起動完了")
    return simulation_app


# def encode_segmentation_video(root_dir: str, start_frame: int, num_frames: int, camera_name: str, output_path: str, env_num: int, trial_num: int) -> None:
#     """セグメンテーション画像から動画をエンコード"""
#     frame_name_pattern = "{camera_name}_semantic_segmentation_trial_{trial_num}_tile_{env_num}_step_{frame_idx}.png"

#     # 最初のフレームからサイズを取得
#     first_frame_path = os.path.join(root_dir, frame_name_pattern.format(
#         camera_name=camera_name, trial_num=trial_num, env_num=env_num, frame_idx=start_frame))
#     if not os.path.exists(first_frame_path):
#         raise ValueError(f"First frame not found: {first_frame_path}")

#     first_frame = cv2.imread(first_frame_path)
#     height, width = first_frame.shape[:2]

#     # VideoWriterを初期化
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, 24.0, (width, height))

#     # フレームを読み込んで書き込み
#     for frame_idx in range(start_frame, start_frame + num_frames):
#         frame_path = os.path.join(root_dir, frame_name_pattern.format(
#             camera_name=camera_name, trial_num=trial_num, env_num=env_num, frame_idx=frame_idx))
#         if os.path.exists(frame_path):
#             frame = cv2.imread(frame_path)
#             out.write(frame)
#         else:
#             print(f"Warning: Frame not found: {frame_path}")

#     out.release()
#     print(f"Segmentation video saved to: {output_path}")


def run_video_preprocessing():
    try:
        from video_encoding import get_video_encoding_interface
        print("✓ video_encodingモジュールが利用可能")
    except ImportError as e:
        print(f"✗ video_encodingモジュールが見つかりません: {e}")
        return

    # notebook_utilsから関数をインポート
    try:
        from notebook_utils import encode_video, get_env_trial_frames
        print("✓ notebook_utilsモジュールが利用可能")
    except ImportError as e:
        print(f"✗ notebook_utilsモジュールが見つかりません: {e}")
        return

    # パラメータ設定
    VIDEO_LENGTH = 226
    camera = "table_high_cam"
    # camera = "table_cam"
    print(f"カメラ: {camera}")
    print(f"動画長: {VIDEO_LENGTH}フレーム")

    # フレーム情報を取得
    try:
        env_trial_frames = get_env_trial_frames(OUTPUT_DIR, camera, 10)
        print(f"見つかったtrial: {env_trial_frames}")
    except Exception as e:
        print(f"フレーム情報の取得に失敗: {e}")
        return

    # 各trialを処理
    for env_num, trial_nums in env_trial_frames.items():
        for trial_num, (start_frame, end_frame) in trial_nums.items():
            trial_length = end_frame - start_frame + 1
            if trial_length < VIDEO_LENGTH:
                print(f"\nSkipping Trial {trial_num}: Too short ({trial_length} frames)")
                continue

            video_start = max(start_frame, end_frame - VIDEO_LENGTH + 1)

            # 1. シェーディング適用済み動画（元の処理）
            shaded_video_filepath = os.path.join(OUTPUT_DIR, f"shaded_segmentation_{camera}_trial_{trial_num}_tile_{env_num}.mp4")

            try:
                print(f"\nProcessing trial {trial_num}...")
                print(f"  Start frame: {video_start}")
                print(f"  End frame: {video_start + VIDEO_LENGTH - 1}")
                print(f"  Creating shaded segmentation video...")
                encode_video(OUTPUT_DIR, video_start, VIDEO_LENGTH, 
                            camera, shaded_video_filepath, env_num, trial_num)
                print(f"✓ Successfully created: {shaded_video_filepath}")

            except Exception as e:
                print(f"✗ Error processing shaded video for trial {trial_num}: {str(e)}")
                import traceback
                traceback.print_exc()

            # # 2. セグメンテーション動画（新規追加）
            # segmentation_video_filepath = os.path.join(OUTPUT_DIR, f"segmentation_{camera}_trial_{trial_num}_tile_{env_num}.mp4")

            # try:
            #     print(f"  Creating segmentation video...")
            #     encode_segmentation_video(OUTPUT_DIR, video_start, VIDEO_LENGTH, 
            #                             camera, segmentation_video_filepath, env_num, trial_num)
            #     print(f"✓ Successfully created: {segmentation_video_filepath}")
            # except Exception as e:
            #     print(f"✗ Error processing segmentation video for trial {trial_num}: {str(e)}")
            #     import traceback
            #     traceback.print_exc()

            # print(f"  Trial {trial_num} processing completed.\n")


def main():
    """メイン関数"""
    print("=== Isaac Sim Video Preprocessing ===")

    try:
        simulation_app = start_isaac_sim()
        import gymnasium as gym
        import numpy as np
        import random
        import torch
        import isaaclab_mimic.envs  # noqa: F401
        import isaaclab_tasks  # noqa: F401

        run_video_preprocessing()
        print("\n=== 処理完了 ===")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            if 'simulation_app' in locals():
                simulation_app.close()
                print("Isaac Simを終了しました")
        except:
            pass


if __name__ == "__main__":
    main()
