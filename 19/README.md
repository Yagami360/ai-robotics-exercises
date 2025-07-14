# Isaac Lab Mimic を使用して、シミュレーター環境上での遠隔操作（Teleoperation）で作成した片腕マニピュレーターロボット（Franka）用の少数の学習用データセットから大量の学習用データセットを自動作成する

[Isaac Sim & Lab のシミュレーター環境上で遠隔操作（Teleoperation）により片腕マニピュレーターロボット（Franka）を操作しながら学習用データセットを作成する](18/README.md) で作成した学習用データセットは、シミュレーター上の手動操作によって作成するので、学習用データセットのレコード数が少ない問題がある。

Isaac Lab Mimic を使用すれば、この手動操作で作成した少数の学習用データセットから、大量の学習用データセットを自動生成することができる

## 方法

1. シミュレーター上の遠隔操作（Teleoperation）で作成したデータセットに（自動データ生成のための）アノテーションを付与する

    まず Isaac Lab Mimic のスクリプトを使用して、シミュレーター上の遠隔操作（Teleoperation）で作成したデータセットに、自動データ生成のためのアノテーションを付与する必要がある。

    - state-based policy を使用する場合

        ```bash
        ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
            --device cuda \
            --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
            --auto \
            --input_file ../datasets/teleop_franka_demo/dataset.hdf5 \
            --output_file ../datasets/teleop_franka_demo/annotated_dataset.hdf5
        ```
        - `--auto`: 自動アノテーション

        アノテーションが付与されたデータセットの具体的な中身は以下のようになっており、`obs/datagen_info` にアノテーションデータが付与されている。

        ```bash
        (base) sakai@sakai-gpu-dev-2:~/personal-repositories/ai-robotics-exercises/datasets/teleop_franka_demo$ h5ls -r annotated_dataset.hdf5
        /                        Group
        /data                    Group
        /data/demo_0             Group
        /data/demo_0/actions     Dataset {236, 7}
        /data/demo_0/initial_state Group
        /data/demo_0/initial_state/articulation Group
        /data/demo_0/initial_state/articulation/robot Group
        /data/demo_0/initial_state/articulation/robot/joint_position Dataset {1, 9}
        /data/demo_0/initial_state/articulation/robot/joint_velocity Dataset {1, 9}
        /data/demo_0/initial_state/articulation/robot/root_pose Dataset {1, 7}
        /data/demo_0/initial_state/articulation/robot/root_velocity Dataset {1, 6}
        /data/demo_0/initial_state/rigid_object Group
        /data/demo_0/initial_state/rigid_object/cube_1 Group
        /data/demo_0/initial_state/rigid_object/cube_1/root_pose Dataset {1, 7}
        /data/demo_0/initial_state/rigid_object/cube_1/root_velocity Dataset {1, 6}
        /data/demo_0/initial_state/rigid_object/cube_2 Group
        /data/demo_0/initial_state/rigid_object/cube_2/root_pose Dataset {1, 7}
        /data/demo_0/initial_state/rigid_object/cube_2/root_velocity Dataset {1, 6}
        /data/demo_0/initial_state/rigid_object/cube_3 Group
        /data/demo_0/initial_state/rigid_object/cube_3/root_pose Dataset {1, 7}
        /data/demo_0/initial_state/rigid_object/cube_3/root_velocity Dataset {1, 6}
        /data/demo_0/obs         Group
        /data/demo_0/obs/actions Dataset {236, 7}
        /data/demo_0/obs/cube_orientations Dataset {236, 12}
        /data/demo_0/obs/cube_positions Dataset {236, 9}
        /data/demo_0/obs/datagen_info Group
        /data/demo_0/obs/datagen_info/eef_pose Group
        /data/demo_0/obs/datagen_info/eef_pose/franka Dataset {236, 4, 4}
        /data/demo_0/obs/datagen_info/object_pose Group
        /data/demo_0/obs/datagen_info/object_pose/cube_1 Dataset {236, 4, 4}
        /data/demo_0/obs/datagen_info/object_pose/cube_2 Dataset {236, 4, 4}
        /data/demo_0/obs/datagen_info/object_pose/cube_3 Dataset {236, 4, 4}
        /data/demo_0/obs/datagen_info/subtask_term_signals Group
        /data/demo_0/obs/datagen_info/subtask_term_signals/grasp_1 Dataset {236}
        /data/demo_0/obs/datagen_info/subtask_term_signals/grasp_2 Dataset {236}
        /data/demo_0/obs/datagen_info/subtask_term_signals/stack_1 Dataset {236}
        /data/demo_0/obs/datagen_info/target_eef_pose Group
        /data/demo_0/obs/datagen_info/target_eef_pose/franka Dataset {236, 4, 4}
        /data/demo_0/obs/eef_pos Dataset {236, 3}
        /data/demo_0/obs/eef_quat Dataset {236, 4}
        /data/demo_0/obs/gripper_pos Dataset {236, 2}
        /data/demo_0/obs/joint_pos Dataset {236, 9}
        /data/demo_0/obs/joint_vel Dataset {236, 9}
        /data/demo_0/obs/object  Dataset {236, 39}
        /data/demo_0/states      Group
        /data/demo_0/states/articulation Group
        /data/demo_0/states/articulation/robot Group
        /data/demo_0/states/articulation/robot/joint_position Dataset {236, 9}
        /data/demo_0/states/articulation/robot/joint_velocity Dataset {236, 9}
        /data/demo_0/states/articulation/robot/root_pose Dataset {236, 7}
        /data/demo_0/states/articulation/robot/root_velocity Dataset {236, 6}
        /data/demo_0/states/rigid_object Group
        /data/demo_0/states/rigid_object/cube_1 Group
        /data/demo_0/states/rigid_object/cube_1/root_pose Dataset {236, 7}
        /data/demo_0/states/rigid_object/cube_1/root_velocity Dataset {236, 6}
        /data/demo_0/states/rigid_object/cube_2 Group
        /data/demo_0/states/rigid_object/cube_2/root_pose Dataset {236, 7}
        /data/demo_0/states/rigid_object/cube_2/root_velocity Dataset {236, 6}
        /data/demo_0/states/rigid_object/cube_3 Group
        /data/demo_0/states/rigid_object/cube_3/root_pose Dataset {236, 7}
        /data/demo_0/states/rigid_object/cube_3/root_velocity Dataset {236, 6}
        /data/demo_1             Group
        /data/demo_1/actions     Dataset {233, 7}
        /data/demo_1/initial_state Group
        ...
        ```

        より詳細には、以下のデータが付与されている

        - サブタスク終了信号
            ```bash
            /data/demo_X/obs/datagen_info/subtask_term_signals/
            ├── grasp_1    Dataset {N}  # 最初のキューブを掴むサブタスクの終了信号
            ├── grasp_2    Dataset {N}  # 2番目のキューブを掴むサブタスクの終了信号
            └── stack_1    Dataset {N}  # キューブを積み重ねるサブタスクの終了信号
            ```

            このデータセットのタスク（オブジェクトを掴んで指定の位置に移動させる）を構成するサブタスク（オブジェクトを掴む・オブジェクトを移動する・オブジェクトを離す）の終了信号（False: サブタスクがまだ完了していない、True: サブタスクが完了した）のアノテーション

        - データ生成用情報
            ```bash
            /data/demo_X/obs/datagen_info/
            ├── eef_pose/franka                    Dataset {N, 4, 4}  # エンドエフェクタの姿勢
            ├── object_pose/
            │   ├── cube_1                         Dataset {N, 4, 4}  # キューブ1の姿勢
            │   ├── cube_2                         Dataset {N, 4, 4}  # キューブ2の姿勢
            │   └── cube_3                         Dataset {N, 4, 4}  # キューブ3の姿勢
            ├── subtask_term_signals/              # サブタスク終了信号（上記）
            └── target_eef_pose/franka             Dataset {N, 4, 4} 
            ```

    - visuomotor policy を使用する場合

        ```bash
        ```

1. （自動データ生成のための）アノテーション付与した少数の学習用データセットから大量の学習用データセットを作成する

    Isaac Lab Mimic のスクリプトを使用して、上記自動データ生成のためのアノテーションを付与した少数の学習用データセットから大量の学習用データセットを自動的に作成する

    - state-based policy を使用する場合

        ```bash
        ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
            --device cuda --headless \
            --num_envs 10 \
            --generation_num_trials 1000 \
            --input_file ../datasets/teleop_franka_demo/annotated_dataset.hdf5 \
            --output_file ./datasets/teleop_franka_demo/generated_dataset.hdf5
        ```

        - `--generation_num_trials` の値で生成するレコード数を指定できる
