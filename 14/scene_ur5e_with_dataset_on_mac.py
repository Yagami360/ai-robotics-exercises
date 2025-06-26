import argparse
import genesis as gs
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def run_simulation_from_dataset(scene, camera, robot, dataset, dofs_idx):
    # エピソードの各ステップを実行
    for step_idx, step_data in enumerate(dataset):
        print(f"ステップ {step_idx + 1}/{len(dataset)}")

        # アクションを取得（データセットは7次元、UR5eは6DOF）
        dataset_action = step_data["action"].numpy()
        action = dataset_action[0:6]
        # action = dataset_action[1:7]

        # アクションをスケーリング（データセットの値は小さすぎる可能性がある）
        # データセットのアクションは通常、小さな相対的な変化量
        scaled_action = action * 10.0

        # ロボットの状態を取得
        robot_state = step_data["observation.state"].numpy()
        print(f"  アクション（データセット）: {dataset_action}")
        print(f"  アクション: {action}")
        print(f"  スケール済みアクション: {scaled_action}")
        print(f"  ロボット状態: {robot_state}")

        # ロボットにアクションを適用
        robot.set_dofs_position(scaled_action, dofs_idx)

        # シミュレーションステップ
        scene.step()

        # カメラの位置を更新（ロボットの動きに合わせて）
        camera.set_pose(
            pos=(3.0 * np.sin(step_idx / 30), 3.0 * np.cos(step_idx / 30), 2.5),
        )
        camera.render()

        print("エピソード完了")
        camera.stop_recording(save_to_filename=f'video-dataset-action0-6-scale-{scale}.mp4', fps=30)
        scene.viewer.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--dataset_id', type=str, default='lerobot/berkeley_autolab_ur5')
    parser.add_argument('--resolution', type=tuple, default=(1280, 960))
    parser.add_argument('--camera_pos', type=tuple, default=(0, -3.5, 2.5))
    parser.add_argument('--camera_lookat', type=tuple, default=(0.0, 0.0, 0.5))
    parser.add_argument('--camera_fov', type=float, default=30)
    parser.add_argument('--show_viewer', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f'{key}: {value}')

    # -------------------------------------
    # initialize Genesis
    # -------------------------------------
    if args.device == 'cpu':
        gs.init(backend=gs.cpu)
    elif args.device == 'cuda':
        gs.init(backend=gs.gpu)
    else:
        raise ValueError(f'Invalid device: {args.device}')

    # -------------------------------------
    # create scene
    # -------------------------------------
    scene = gs.Scene(
        show_viewer = args.show_viewer,
        viewer_options = gs.options.ViewerOptions(
            res           = args.resolution,
            camera_pos    = args.camera_pos,
            camera_lookat = args.camera_lookat,
            camera_fov    = args.camera_fov,
            max_FPS       = 60,
        ),
        vis_options = gs.options.VisOptions(
            show_world_frame = True,
            world_frame_size = 1.0,
            show_link_frame  = False,
            show_cameras     = False,
            plane_reflection = True,
            ambient_light    = (0.1, 0.1, 0.1),
        ),
        sim_options = gs.options.SimOptions(
            dt = 0.01,
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    # -------------------------------------
    # add camera
    # -------------------------------------
    camera = scene.add_camera(
        res    = args.resolution,
        pos    = args.camera_pos,
        lookat = args.camera_lookat,
        fov    = args.camera_fov,
        GUI    = False,
    )

    # -------------------------------------
    # add objects (entities)
    # -------------------------------------
    plane = scene.add_entity(gs.morphs.Plane())

    # Add robot (UR5e)
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file='xml/universal_robots_ur5e/ur5e.xml',
            pos   = (0.0, 0.0, 0.0),
            euler = (0, 0, 0),
        ),
    )
    print("robot: ", robot)

    # -------------------------------------
    # build scene
    # -------------------------------------
    scene.build()

    # -------------------------------------
    # set robot joint control parameters
    # -------------------------------------
    joint_names = [
        # 'base_joint', # 固定ジョイントのため除外
        'shoulder_pan',
        'shoulder_lift',
        'elbow',
        'wrist_1',
        'wrist_2',
        'wrist_3',
        # 'ee_virtual_link_joint', # 固定ジョイントのため除外
    ]
    dofs_idx = [robot.get_joint(name).dof_idx_local for name in joint_names]
    print("dofs_idx: ", dofs_idx)

    # 位置ゲインの設定
    # 位置ゲイン：ロボットの位置制御において、目標位置と現在位置の差（位置誤差）をどれだけ強く補正するかを決めるパラメータ
    robot.set_dofs_kp(
        kp             = np.array([2000, 2000, 2000, 500, 500, 500]),
        dofs_idx_local = dofs_idx,
    )

    # 速度ゲインの設定
    # 速度ゲイン：ロボットの速度制御において、目標速度と現在速度の差（速度誤差）をどれだけ強く補正するかを決めるパラメータ
    robot.set_dofs_kv(
        kv             = np.array([100, 100, 100, 25, 25, 25]),
        dofs_idx_local = dofs_idx,
    )

    # 安全のための力の範囲設定
    robot.set_dofs_force_range(
        lower          = np.array([-150, -150, -150, -28, -28, -28]),
        upper          = np.array([ 150,  150,  150,  28,  28,  28]),
        dofs_idx_local = dofs_idx,
    )

    # -------------------------------------
    # load dataset
    # -------------------------------------
    print(f"データセット '{args.dataset_id}' を読み込み中...")
    dataset = LeRobotDataset(args.dataset_id, episodes=[args.episode])
    print(f"dataset: {dataset}")

    # -------------------------------------
    # run simulation from dataset
    # -------------------------------------
    print(f"エピソード {args.episode} の実行を開始...")

    gs.tools.run_in_another_thread(
        fn=run_simulation_from_dataset,
        args=(scene, camera, robot, dataset, dofs_idx)
    )

    camera.start_recording()
    scene.viewer.start()
