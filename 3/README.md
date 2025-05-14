# LeRobot の π0 モデルを gymnasium のシミュレーター環境で推論する（自身で実装した推論スクリプトを使用する場合）

## 使用方法

1. CPU メモリ 16GB 程度, GPU メモリ 32GB 以上（V100, A100など）の GPU インスタンスを用意する

1. LeRobot の π0 をインストールする<br>
    ```sh
    cd ${PROJECT_ROOT}/lerobot
    pip install -e ".[pi0]"
    ```

1. Hugging Face にログインする
    ```sh
    huggingface-cli login ${TOKEN}
    ```

    > 推論時に、π0モデルが内部で `google/paligemma-3b-pt-224` というプライベートリポジトリにアクセスするために必要

1. Hugging Face 上の Paligemma のリポジトリのアクセス権限をリクエストする<br>
    "https://huggingface.co/google/paligemma-3b-pt-224" に移動して、アクセス権限をリクエストする

1. π0 モデルを pusht タスク用にファインチューニングする

    ```sh
    cd lerobot
    python lerobot/scripts/train.py \
        --policy.path=lerobot/pi0 \
        --dataset.repo_id=lerobot/pusht \
        --env.type=pusht \
        --batch_size=2 \
        --num_workers=2 \
        --steps=100000 \
        --policy.device=cuda
    ```
    - `batch_size`, `num_workers`, `steps` は、インスタンス環境に応じて要調整
    - 環境に応じて、`--policy.use_amp=true`, `--policy.device=cpu` を指定する
    - `env.type` : ロボットのタスク
        - `pusht`: ロボットが平面上のオブジェクトをT字型のターゲット位置に押し込むタスク
        - `aloha`: 両腕を使った複雑な操作タスク（物体の把持、移動、操作など）
        - `xarm`: UFactory社のxArmロボットアームを使用する環境で、単腕ロボットによる様々なマニピュレーションタスク

    学習用データセットを pusht 用のデータセットとし、シミュレーター環境も pusht 用に設定し、π0 モデルを pusht タスク用にファインチューニングする

    - `lerobot/pusht` データセットの中身

        ```yml
        {
            # 環境の画像
            'observation.image': tensor([[[1.0000, 0.9725, 0.9725,  ..., 0.9725, 0.9725, 1.0000],
                [0.9725, 0.9098, 0.9098,  ..., 0.9098, 0.9098, 0.9725],
                [0.9725, 0.9098, 0.9725,  ..., 1.0000, 0.9098, 0.9725],
                ...,
                [1.0000, 0.9725, 0.9725,  ..., 0.9725, 0.9725, 1.0000]]]),
            # ロボットの x, y 位置
            'observation.state': tensor([222.,  97.]),
            # ロボットの次の行動
            'action': tensor([233.,  71.]),
            # エピソードの値
            'episode_index': tensor(0),
            # フレーム（時間ステップ）
            'frame_index': tensor(0),
            # 時間情報
            'timestamp': tensor(0.),
            # エピソードが終了したかどうかのフラグ
            'next.done': tensor(False),
            # エピソードが成功したかどうかのフラグ
            'next.success': tensor(False),
            # データセット内の絶対的なインデックス
            'index': tensor(0),
            # タスクの種類を示すインデックス
            'task_index': tensor(0),
            # ロボットへの制御指示テキスト
            'task': 'Push the T-shaped block onto the T-shaped target.'
        }
        ```

1. gymnasium のシミュレーターを使用した π0 モデルの推論スクリプトを実装する

    ```python
    import os

    import torch
    import numpy
    import imageio

    import gym_pusht  # noqa: F401
    import gymnasium as gym

    import lerobot
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    load_checkpoint_dir = "../checkpoints/08-56-36_pusht_pi0/checkpoints/last/pretrained_model"
    output_dir = "outputs/eval/pi0"
    os.makedirs(output_dir, exist_ok=True)

    # Select your device
    device = "cuda"
    # device = "cpu"

    # Define the policy
    policy = PI0Policy.from_pretrained(
        load_checkpoint_dir,
        strict=False,
    )
    print("Policy config:", vars(policy.config))

    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent. The environment
    # also automatically stops running after 300 interactions/steps.
    env = gym.make(
        "gym_pusht/PushT-v0",
        # 観測データ（observation） は、ロボットの x,y位置（pos） + 環境の画像（pixels）
        obs_type="pixels_agent_pos",
        max_episode_steps=300,
    )

    # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # produced by the environment
    print("policy.config.input_features:", policy.config.input_features)
    print("env.observation_space:", env.observation_space)

    # Similarly, we can check that the actions produced by the policy will match the actions expected by the
    # environment
    print("policy.config.output_features:", policy.config.output_features)
    print("env.action_space:", env.action_space)

    # Reset the policy and environments to prepare for rollout
    policy.reset()
    numpy_observation, info = env.reset(seed=42)

    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    frames = []

    # Render frame of the initial state
    frames.append(env.render())

    step = 0
    done = False
    while not done:
        # ロボットの x, y 位置
        state = torch.from_numpy(numpy_observation["agent_pos"]).to(device)
        state = state.to(torch.float32)
        state = state.unsqueeze(0)

        # 環境の画像
        image = torch.from_numpy(numpy_observation["pixels"]).to(device)
        image = image.to(torch.float32) / 255
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)

        # π0 モデルのポリシー（行動方策）が期待する形式に合わせて観測データ（observation）を構成
        observation = {
            # ロボットの状態
            "observation.state": state,
            # 環境の画像
            "observation.image": image,
            # ロボットへの制御指示テキスト
            # `lerobot/pusht` の学習用データセットと同じ内容のテキストにする
            "task": ["Push the T-shaped block onto the T-shaped target."]
        }

        # π0 モデルの行動方策に基づき、次の行動を推論
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Prepare the action for the environment
        action_np = action.squeeze(0).to("cpu").numpy()

        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, info = env.step(action_np)
        print(f"{step=} {reward=} {terminated=}")

        # Keep track of all the rewards and frames
        rewards.append(reward)
        frames.append(env.render())

        # The rollout is considered done when the success state is reached (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1

    if terminated:
        print("Success!")
    else:
        print("Failure!")

    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video.
    video_path = os.path.join(output_dir, "frames.mp4")
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

    print(f"Video of the evaluation is available in '{video_path}'.")
    ```

    ポイントは、以下の通り

    - xxx

1. 推論スクリプトを実行する

    ```sh
    python eval.py
    ```

    以下のような出力が得られる

    - 学習ステップ数: 1000 のファインチューニングモデルで推論した場合<br>
        https://github.com/user-attachments/assets/26e729ea-6a1d-46f4-ac49-0f64ab366e54

    - 学習ステップ数: 10000 のファインチューニングモデルで推論した場合<br>
        https://github.com/user-attachments/assets/2db2f5c9-026e-4669-b6b7-bbdea3e61276

    - 学習ステップ数: 100000 のファインチューニングモデルで推論した場合<br>
