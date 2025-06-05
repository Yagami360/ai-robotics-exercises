# Issac Sim & Labs のシミュレーター環境上でヒューマノイドロボット（GR1）をファインチューニングした Isaac-GR00T モデルで推論させながら動かす

1. Isaac Sim & Labs をインストールする

    「[Issac Sim & Labs の空シミュレーション環境を起動する](https://github.com/Yagami360/ai-robotics-exercises/blob/master/7/README.md#vnc-%E3%82%92%E4%BD%BF%E7%94%A8%E3%81%97%E3%81%A6-ubuntu-%E3%82%B5%E3%83%BC%E3%83%90%E3%83%BC%E3%81%AA%E3%81%A9%E3%81%AE%E9%9D%9Egui%E7%92%B0%E5%A2%83%E3%81%A7%E5%8B%95%E3%81%8B%E3%81%99%E5%A0%B4%E5%90%88)」記載の方法で、Isaac Sim & Labs をインストールする

1. Isaac-GR00T をインストールする

    「[LeRobot の事前学習済み Isaac-GR00T モデルに対してデモ用データセットで推論を行なう](https://github.com/Yagami360/ai-robotics-exercises/blob/master/6/README.md)」記載の方法で、Isaac-GR00T をインストールする

    > 今回の例では、`isaac-labs` の conda 環境にIsaac-GR00T をインストール

1. Isaac-GR00T モデルをファインチューニングする

    デモ用データセット（`Isaac-GR00T/demo_data/robot_sim.PickNPlace`）で Isaac-GR00T モデルをファインチューニングする

    このデモ用データセットは、以下の動画のように、ヒューマノイドロボット（GR1 ?）が物を掴んで皿に置くというタスクのデータセットになっている

    https://github.com/user-attachments/assets/70b908b3-36ce-43e7-9df4-cc5cff9559e0

    Isaac-GR00T モデルのファインチューニングは、以下のコマンドで簡単に実行可能

    ```bash
    mkdir -p checkpoints/gr00t
    cd Isaac-GR00T
    python scripts/gr00t_finetune.py --dataset-path ./demo_data/robot_sim.PickNPlace --num-gpus 1 --output-dir ../checkpoints/gr00t
    ```

    この学習スクリプトはデフォルトでは、A100, RTX3000番台 などの Ampere 世代の GPU のみ実行可能になっている。T4,V100 のような GPU でも動かすには、`scripts/gr00t_finetune.py` の `tf32` を True -> False に変更すればよい（但し GPU メモリは 36GB 程度必要なので、複数GPUで動かす必要あり）

    - GPUメモリ使用量（学習時は36GB程度。推論時は T4 等のサイズでも動く）

        ```bash
        Every 2.0s: nvidia-smi                                                   sakai-gpu-dev-2: Wed Jun  4 10:18:47 2025

        Wed Jun  4 10:18:47 2025
        +-----------------------------------------------------------------------------------------+
        | NVIDIA-SMI 575.51.03              Driver Version: 575.51.03      CUDA Version: 12.9     |
        |-----------------------------------------+------------------------+----------------------+
        | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
        |                                         |                        |               MIG M. |
        |=========================================+========================+======================|
        |   0  NVIDIA A100-SXM4-40GB          Off |   00000000:00:04.0 Off |                    0 |
        | N/A   40C    P0            111W /  400W |   36160MiB /  40960MiB |     87%      Default |
        |                                         |                        |             Disabled |
        +-----------------------------------------+------------------------+----------------------+

        +-----------------------------------------------------------------------------------------+
        | Processes:                                                                              |
        |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
        |        ID   ID                                                               Usage      |
        |=========================================================================================|
        |    0   N/A  N/A            2065      C   python                                36148MiB |
        +-----------------------------------------------------------------------------------------+
        ```

    - 損失関数のグラフ

        <img width="800" alt="Image" src="https://github.com/user-attachments/assets/203888d4-9802-47c4-ac5b-b9c742493583" />

1. Isaac Sim & Labs のシミュレーターを使用した推論コードを実装する

    今回の例では、[GR1T2]() というヒューマノイドロボットが、（学習用データセットと同じ）物を掴んで皿に置くというシミュレーター環境（シーン）にしている

    [`eval_isaaclab_scene_gr1t2.py`](./eval_isaaclab_scene_gr1t2.py)

1. VNC サーバーを起動する

    「[Issac Sim & Labs の空シミュレーション環境を起動する](https://github.com/Yagami360/ai-robotics-exercises/blob/master/7/README.md#vnc-%E3%82%92%E4%BD%BF%E7%94%A8%E3%81%97%E3%81%A6-ubuntu-%E3%82%B5%E3%83%BC%E3%83%90%E3%83%BC%E3%81%AA%E3%81%A9%E3%81%AE%E9%9D%9Egui%E7%92%B0%E5%A2%83%E3%81%A7%E5%8B%95%E3%81%8B%E3%81%99%E5%A0%B4%E5%90%88)」記載の方法で、VNC 環境を構築した上で、以下のコマンドを実行する

    ```bash
    vncserver :1 -geometry 1280x720 -depth 16 -localhost no -SecurityTypes VncAuth -SendCutText=0 -AcceptCutText=0 -AcceptPointerEvents=1 -AcceptKeyEvents=1
    ```

1. シミュレーターを起動する

    ```bash
    conda activate isaac-labs
    python eval_isaaclab_scene_gr1t2.py --model_path ../checkpoints/gr00t/checkpoint-3000
    ```

    > 現状のコードでは、シミュレーター起動後に GUIからロボットのカメラの回転角を (0.0, 0.0, -90)に手動で変更する必要あり
    > <img width="500" alt="Image" src="https://github.com/user-attachments/assets/8a718584-814e-46ae-babd-1e6f655b4e76" />


    今回は、以下のようにロボットがうまく物を掴めない結果になった。<br>
    これは、学習用データセットのロボットとシミュレーター環境上のロボット（GR1T2）のロボットの種類が異なることが原因と考えられる。実際に学習用データセットにおける手の関節の次元数は 6 次元だが、シミュレーター環境上の GR1T2 ロボットの手の関節の次元数は 12 次元だった

https://github.com/user-attachments/assets/6453bbf7-5f38-410a-9c55-8d06c762b4d1


<!--
1. pinocchio をインストールする
    `[Isaac-PickPlace-GR1T2-Abs-v0](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_gr1t2_env_cfg.py)` の環境では、pinocchio を使用しているので、インストールする

    ```bash
    conda install -c conda-forge pinocchio -y
    ```
    > pinocchio: ロボットの運動学と動力学を計算するためのライブラリ

    pinocchio インストール後に Issac Lab の numpy バージョンと不整合が発生した場合は、以下のコマンドも実行する

    ```bash
    conda install numpy=1.25.0 -y
    ```

1. 利用可能な環境のリストを確認する

    ```bash
    python scripts/environments/list_envs.py
    ```
-->
