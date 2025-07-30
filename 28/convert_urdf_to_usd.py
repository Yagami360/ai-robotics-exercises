#!/usr/bin/env python3
"""
URDFファイルをUSDファイルに変換するスクリプト

このスクリプトは、SO-ARM101ロボットのURDFファイルをUSDファイルに変換します。
Isaac Simで使用するために必要です。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def convert_urdf_to_usd(urdf_path, usd_path):
    """
    URDFファイルをUSDファイルに変換する
    
    Args:
        urdf_path (str): URDFファイルのパス
        usd_path (str): 出力USDファイルのパス
    
    Returns:
        bool: 変換が成功したかどうか
    """
    try:
        # Isaac SimのURDF to USD変換ツールを使用
        cmd = [
            "python", "-m", "omni.isaac.core.utils.urdf",
            "--input", urdf_path,
            "--output", usd_path
        ]
        
        print(f"URDFファイルをUSDに変換中: {urdf_path} -> {usd_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("変換が完了しました")
            return True
        else:
            print(f"変換エラー: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"変換中にエラーが発生しました: {e}")
        return False


def check_urdf_file(urdf_path):
    """
    URDFファイルの存在と内容をチェック
    
    Args:
        urdf_path (str): URDFファイルのパス
    
    Returns:
        bool: ファイルが有効かどうか
    """
    if not os.path.exists(urdf_path):
        print(f"エラー: URDFファイルが見つかりません: {urdf_path}")
        return False
    
    # ファイルサイズをチェック
    file_size = os.path.getsize(urdf_path)
    if file_size == 0:
        print(f"エラー: URDFファイルが空です: {urdf_path}")
        return False
    
    print(f"URDFファイルを確認しました: {urdf_path} ({file_size} bytes)")
    return True


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="URDFファイルをUSDファイルに変換するスクリプト"
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="../assets/SO101/so101_new_calib.urdf",
        help="入力URDFファイルのパス"
    )
    parser.add_argument(
        "--usd_path",
        type=str,
        default="so101_robot.usd",
        help="出力USDファイルのパス"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存のUSDファイルを上書きする"
    )
    
    args = parser.parse_args()
    
    # URDFファイルをチェック
    if not check_urdf_file(args.urdf_path):
        sys.exit(1)
    
    # 出力ディレクトリを作成
    usd_dir = os.path.dirname(args.usd_path)
    if usd_dir and not os.path.exists(usd_dir):
        os.makedirs(usd_dir)
        print(f"出力ディレクトリを作成しました: {usd_dir}")
    
    # 既存のUSDファイルをチェック
    if os.path.exists(args.usd_path) and not args.force:
        print(f"警告: USDファイルが既に存在します: {args.usd_path}")
        print("上書きするには --force オプションを使用してください")
        response = input("上書きしますか？ (y/N): ")
        if response.lower() != 'y':
            print("変換をキャンセルしました")
            sys.exit(0)
    
    # URDFをUSDに変換
    if convert_urdf_to_usd(args.urdf_path, args.usd_path):
        print(f"変換が成功しました: {args.usd_path}")
        
        # 変換されたファイルの情報を表示
        if os.path.exists(args.usd_path):
            file_size = os.path.getsize(args.usd_path)
            print(f"USDファイルサイズ: {file_size} bytes")
    else:
        print("変換に失敗しました")
        sys.exit(1)


if __name__ == "__main__":
    main() 