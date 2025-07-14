#!/usr/bin/env python3
"""
HDF5ファイルを可視化するためのスクリプト
Isaac Lab Mimicで生成されたデータセットを確認できます
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

def explore_hdf5_structure(file_path):
    """HDF5ファイルの構造を探索して表示"""
    print(f"\n=== HDF5ファイル構造: {file_path} ===")
    
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
        
        f.visititems(print_structure)

def visualize_episode_data(file_path, episode_name="demo_0"):
    """特定のエピソードのデータを可視化"""
    print(f"\n=== エピソードデータ: {episode_name} ===")
    
    with h5py.File(file_path, 'r') as f:
        if f"data/{episode_name}" not in f:
            print(f"エピソード {episode_name} が見つかりません")
            available_episodes = list(f['data'].keys())
            print(f"利用可能なエピソード: {available_episodes}")
            return
        
        episode = f[f"data/{episode_name}"]
        
        # 各データセットの情報を表示
        for key, dataset in episode.items():
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
            
            # 数値データの統計情報を表示
            if dataset.dtype.kind in 'fc':  # float or complex
                data = dataset[:]
                print(f"    min: {np.min(data):.4f}, max: {np.max(data):.4f}, mean: {np.mean(data):.4f}")
            
            # 最初の数個の値を表示
            if len(dataset.shape) == 1:
                print(f"    最初の5個の値: {dataset[:5]}")
            elif len(dataset.shape) == 2:
                print(f"    最初の3行3列: {dataset[:3, :3]}")

def plot_actions_over_time(file_path, episode_name="demo_0"):
    """アクションを時間軸でプロット"""
    print(f"\n=== アクションの時間変化: {episode_name} ===")
    
    with h5py.File(file_path, 'r') as f:
        if f"data/{episode_name}" not in f:
            print(f"エピソード {episode_name} が見つかりません")
            return
        
        episode = f[f"data/{episode_name}"]
        
        if 'actions' not in episode:
            print("アクションデータが見つかりません")
            return
        
        actions = episode['actions'][:]
        
        # アクションの次元数を確認
        if len(actions.shape) == 2:
            num_actions = actions.shape[1]
            time_steps = actions.shape[0]
            
            # 各アクション次元を別々のサブプロットで表示
            fig, axes = plt.subplots(num_actions, 1, figsize=(12, 2*num_actions))
            if num_actions == 1:
                axes = [axes]
            
            for i in range(num_actions):
                axes[i].plot(actions[:, i], label=f'Action {i}')
                axes[i].set_title(f'Action Dimension {i}')
                axes[i].set_xlabel('Time Step')
                axes[i].set_ylabel('Action Value')
                axes[i].grid(True)
                axes[i].legend()
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"予期しないアクションの形状: {actions.shape}")

def plot_observations_over_time(file_path, episode_name="demo_0"):
    """観測データを時間軸でプロット"""
    print(f"\n=== 観測データの時間変化: {episode_name} ===")
    
    with h5py.File(file_path, 'r') as f:
        if f"data/{episode_name}" not in f:
            print(f"エピソード {episode_name} が見つかりません")
            return
        
        episode = f[f"data/{episode_name}"]
        
        # 観測データを探す
        obs_keys = [key for key in episode.keys() if key.startswith('obs')]
        
        if not obs_keys:
            print("観測データが見つかりません")
            return
        
        for obs_key in obs_keys:
            obs_data = episode[obs_key][:]
            print(f"  {obs_key}: shape={obs_data.shape}")
            
            # 1次元の観測データをプロット
            if len(obs_data.shape) == 2:
                num_dims = obs_data.shape[1]
                time_steps = obs_data.shape[0]
                
                if num_dims <= 10:  # 次元数が少ない場合のみプロット
                    fig, axes = plt.subplots(num_dims, 1, figsize=(12, 2*num_dims))
                    if num_dims == 1:
                        axes = [axes]
                    
                    for i in range(num_dims):
                        axes[i].plot(obs_data[:, i], label=f'{obs_key} dim {i}')
                        axes[i].set_title(f'{obs_key} - Dimension {i}')
                        axes[i].set_xlabel('Time Step')
                        axes[i].set_ylabel('Value')
                        axes[i].grid(True)
                        axes[i].legend()
                    
                    plt.tight_layout()
                    plt.show()

def check_subtask_annotations(file_path, episode_name="demo_0"):
    """サブタスクアノテーションを確認"""
    print(f"\n=== サブタスクアノテーション: {episode_name} ===")
    
    with h5py.File(file_path, 'r') as f:
        if f"data/{episode_name}" not in f:
            print(f"エピソード {episode_name} が見つかりません")
            return
        
        episode = f[f"data/{episode_name}"]
        
        # サブタスク関連のデータを探す
        subtask_keys = [key for key in episode.keys() if 'subtask' in key.lower()]
        
        if not subtask_keys:
            print("サブタスク関連のデータが見つかりません")
            return
        
        for subtask_key in subtask_keys:
            subtask_data = episode[subtask_key][:]
            print(f"  {subtask_key}: shape={subtask_data.shape}")
            
            # ブール値の場合は変化点を検出
            if subtask_data.dtype == bool or subtask_data.dtype == np.bool_:
                # 0から1への変化点を検出
                if len(subtask_data.shape) == 1:
                    changes = np.diff(subtask_data.astype(int))
                    transition_points = np.where(changes == 1)[0]
                    print(f"    変化点（0→1）: {transition_points}")
                    
                    # 変化点をプロット
                    plt.figure(figsize=(12, 4))
                    plt.plot(subtask_data, label=subtask_key)
                    plt.scatter(transition_points, np.ones_like(transition_points), 
                              color='red', s=50, label='Transition Points')
                    plt.title(f'Subtask Signal: {subtask_key}')
                    plt.xlabel('Time Step')
                    plt.ylabel('Signal Value')
                    plt.legend()
                    plt.grid(True)
                    plt.show()

def main():
    parser = argparse.ArgumentParser(description='HDF5ファイルを可視化')
    parser.add_argument('file_path', type=str, help='HDF5ファイルのパス')
    parser.add_argument('--episode', type=str, default='demo_0', help='可視化するエピソード名')
    parser.add_argument('--explore', action='store_true', help='ファイル構造を探索')
    parser.add_argument('--actions', action='store_true', help='アクションをプロット')
    parser.add_argument('--observations', action='store_true', help='観測データをプロット')
    parser.add_argument('--subtasks', action='store_true', help='サブタスクアノテーションを確認')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"ファイルが見つかりません: {args.file_path}")
        return
    
    # デフォルトですべて実行
    if not any([args.explore, args.actions, args.observations, args.subtasks]):
        args.explore = True
        args.actions = True
        args.observations = True
        args.subtasks = True
    
    if args.explore:
        explore_hdf5_structure(args.file_path)
        visualize_episode_data(args.file_path, args.episode)
    
    if args.actions:
        plot_actions_over_time(args.file_path, args.episode)
    
    if args.observations:
        plot_observations_over_time(args.file_path, args.episode)
    
    if args.subtasks:
        check_subtask_annotations(args.file_path, args.episode)

if __name__ == "__main__":
    main() 