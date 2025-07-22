# ai-robotics-exercises

## Huggingface LeRobot

- [LeRobot のチュートリアルを実行する](1/README.md)
- [LeRobot のデータセットを読み込む](2/README.md)
- [LeRobot の事前学習済み Isaac-GR00T モデルに対してデモ用データセットで推論を行なう](6/README.md)
- [[In-progress] LeRobot のデータセット形式での独自データセットを作成する](12/README.md)
- [Rerun を使用してデータセットの可視化を行う](21/README.md)

- gymnasium (旧 OpenAI Gym) のシミュレーション環境を使用
    - [LeRobot の π0 モデルを gymnasium のシミュレーター環境で推論する（Lerobot 提供の推論スクリプトを使用する場合）](4/README.md)
    - [LeRobot の π0 モデルを gymnasium のシミュレーター環境で推論する（自身で実装した推論スクリプトを使用する場合）](3/README.md)
    - [LeRobot の π0 モデルを gymnasium のシミュレーター環境でファインチューニングする（自身で実装した学習スクリプトを使用する場合）](5/README.md)
    - [[In-progress] LeRobot のVLAモデル（π0など）の学習時のデータオーギュメントを改善し汎化性能を向上させる](6/README.md)

- MuJoCo のシミュレーション環境
    - [MuJoCo のシミュレーション環境を起動する](22/README.md)

- Genesis のシミュレーション環境を使用
    - [Genesis のシミュレーション環境を起動する](13/README.md)
    - [[In-progress] Genesis のシミュレーション環境上でロボットを動かす](14/README.md)
    - [Genesis のシミュレーション環境上で学習済みモデルで推論しながらロボットを動かす（公式チュートリアルのコードを使用）](https://genesis-world.readthedocs.io/ja/latest/user_guide/getting_started/locomotion.html)
    - Genesis のシミュレーション環境上で学習済みモデルで推論しながらロボットを動かす（独自のデータセットでファインチューニングしたモデルを使用）

- NVIDIA Isaac Sim & Lab のシミュレーション環境を使用
    - [Isaac Sim & Lab の空シミュレーション環境を起動する](7/README.md)
    - [Isaac Sim & Lab を使用してロボットが配置されたシーンを作成する](8/README.md)
    - Isaac Sim & Lab の環境を作成する
    - Isaac Sim & Lab の環境を登録する
    - Isaac Sim & Lab のシミュレーター上でモデルを推論する：Isaac Lab 提供コードを利用
    - [[In-progress] Isaac Sim & Lab のシミュレーター環境上で片腕マニピュレーターロボット（Franka）をファインチューニングした Isaac-GR00T モデルで推論させながら動かす](10/README.md)
    - [[In-progress] Isaac Sim & Lab のシミュレーター環境上でヒューマノイドロボット（GR1）をファインチューニングした Isaac-GR00T モデルで推論させながら動かす](9/README.md)
    - [Isaac Sim & Lab のシミュレーター環境上で遠隔操作（Teleoperation）により片腕マニピュレーターロボット（Franka）を操作する](17/README.md)
    - [Isaac Sim & Lab のシミュレーター環境上で遠隔操作（Teleoperation）により片腕マニピュレーターロボット（Franka）を操作しながら学習用データセットを作成する](18/README.md)
    - [Isaac Lab Mimic を使用して、シミュレーター環境上での遠隔操作（Teleoperation）で作成した片腕マニピュレーターロボット（Franka）用の少数の学習用データセットから大量の学習用データセットを自動作成する](19/README.md)
    - [Isaac Lab Mimic を使用して、シミュレーター環境上でヒューマノイドロボット（GR1）用の学習用データセットを作成し、Robomimic を使用してモデルを学習＆推論する](20/README.md)
    - Isaac Lab Mimic を使用して作成したHDF5形式での学習用データセットを LeRobot 形式での学習用データセットに変換する
    - NVIDIA Cosmos
        - [NVIDIA Cosmos の概要](23/)
        - [Cosmos Predict を使用して（物理法則が考慮されたフォトリアリスティックな）動画生成を行なう](24/)
        - Cosmos Predict を使用して世界基盤モデルの事後学習を行い生成される合成データの品質を向上させる
        - [Cosmos Transfer を使用してオブジェクトの色や質感などが変化したフォトリアリスティックな動画生成を行なう](25/)
        - [Cosmos Transfer を使用してデータ拡張を行なう](26/)
    - Isaac GR00T blueprint （Isaac Lab Mimic + Cosmos Transfer）を使用して学習用データセット作成とデータ拡張を行なう
