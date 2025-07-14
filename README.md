# ai-robotics-exercises

## Huggingface LeRobot

- [LeRobot のチュートリアルを実行する](1/README.md)
- [LeRobot のデータセットを読み込む](2/README.md)
- [LeRobot の事前学習済み Isaac-GR00T モデルに対してデモ用データセットで推論を行なう](6/README.md)
- [[In-progress] LeRobot のデータセット形式での独自データセットを作成する](12/README.md)


- gymnasium (旧 OpenAI Gym) のシミュレーション環境を使用
    - [LeRobot の π0 モデルを gymnasium のシミュレーター環境で推論する（Lerobot 提供の推論スクリプトを使用する場合）](4/README.md)
    - [LeRobot の π0 モデルを gymnasium のシミュレーター環境で推論する（自身で実装した推論スクリプトを使用する場合）](3/README.md)
    - [LeRobot の π0 モデルを gymnasium のシミュレーター環境でファインチューニングする（自身で実装した学習スクリプトを使用する場合）](5/README.md)
    - [[In-progress] LeRobot のVLAモデル（π0など）の学習時のデータオーギュメントを改善し汎化性能を向上させる](6/README.md)

- Genesis のシミュレーション環境を使用
    - [Genesis のシミュレーション環境を起動する](13/README.md)
    - [[In-progress] Genesis のシミュレーション環境上でロボットを動かす](14/README.md)
    - [Genesis のシミュレーション環境上で学習済みモデルで推論しながらロボットを動かす（公式チュートリアルのコードを使用）](https://genesis-world.readthedocs.io/ja/latest/user_guide/getting_started/locomotion.html)
    - Genesis のシミュレーション環境上で学習済みモデルで推論しながらロボットを動かす（独自のデータセットでファインチューニングしたモデルを使用）

- Isaac Labs のシミュレーション環境を使用
    - [Isaac Sim & Labs の空シミュレーション環境を起動する](7/README.md)
    - [Isaac Sim & Labs を使用してロボットが配置されたシーンを作成する](8/README.md)
    - Isaac Sim & Labs の環境を作成する
    - Isaac Sim & Labs の環境を登録する
    - Isaac Sim & Labs のシミュレーター上でモデルを推論する：Isaac Labs 提供コードを利用
    - [[In-progress] Isaac Sim & Labs のシミュレーター環境上で片腕マニピュレーターロボット（Franka）をファインチューニングした Isaac-GR00T モデルで推論させながら動かす](10/README.md)
    - [[In-progress] Isaac Sim & Labs のシミュレーター環境上でヒューマノイドロボット（GR1）をファインチューニングした Isaac-GR00T モデルで推論させながら動かす](9/README.md)
    - [Isaac Sim & Lab のシミュレーター環境上で遠隔操作（Teleoperation）により片腕マニピュレーターロボット（Franka）を操作する](17/README.md)
    - [Isaac Sim & Lab のシミュレーター環境上で遠隔操作（Teleoperation）により片腕マニピュレーターロボット（Franka）を操作しながら学習用データセットを作成する](18/README.md)
    - [Isaac Lab Mimic を使用して、シミュレーター環境上での遠隔操作（Teleoperation）で作成した片腕マニピュレーターロボット（Franka）用の少数の学習用データセットから大量の学習用データセットを自動作成する](19/README.md)
    - [Isaac Lab Mimic を使用して、シミュレーター環境上でヒューマノイドロボット（GR1）用の学習用データセットを作成し、Robomimic を使用してモデルを学習＆推論する](20/README.md)
    - Isaac Lab Mimic を使用して作成したHDF5形式での学習用データセットを LeRobot 形式での学習用データセットに変換する
